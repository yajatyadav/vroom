import logging
import math
from multiprocessing.pool import ThreadPool
import os
from pathlib import Path
from typing import Iterable, Optional
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from projectaria_tools.projects.aea import (
    AriaEverydayActivitiesDataPathsProvider, 
    AriaEverydayActivitiesDataProvider)
from projectaria_tools.core import calibration, mps
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.sophus import SE3
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.calibration import distort_by_calibration

from torchvision.datasets._optical_flow import _read_flo
from tqdm import tqdm

from anycam.common.io.io import load_array, load_flow
from anycam.datasets.common import get_flow_selector, get_target_size_and_crop, process_depth, process_flow, process_img, process_proj


logger = logging.getLogger(__name__)


RGB_STREAM_ID = StreamId("214-1")


def transform3x4to4x4(m):
    return np.concatenate([m, np.array([[0., 0., 0., 1.]])], axis=0)


XY_FLIP = np.array([
    [0, 1, 0, 0],
    [-1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])


def get_target_size_and_crop_custom(target_size, image_size, invalid_sides=0):
    extra_crop = math.ceil(target_size * invalid_sides)
    expaned_target_size = target_size + (2 * extra_crop)

    if target_size is None:
        return None, None
    elif not isinstance(expaned_target_size, Iterable): # Resize the shorter side to target size and center crop to get a square
        resize_to = (expaned_target_size, expaned_target_size)

        crop = (extra_crop, extra_crop, target_size, target_size)
    else:
        raise NotImplementedError()

    return resize_to, crop



class AriaEADataset(Dataset):
    NAME = "Aria Everyday Activities"

    def __init__(
            self, 
            data_path: str, 
            split_path: Optional[str],
            image_size: Optional[tuple] = None,
            frame_count: int = 2,
            keyframe_offset: int = 0,
            dilation: int = 1,
            return_depth: bool = False,
            full_size_depth: bool = False,
            return_flow: bool = False,
            preprocessed_path: Optional[str] = None,
            selected_sequences: Optional[list] = None,
            flow_selector=None,
            index_selector=None,
            sequence_sampler=None,
        ):
        
        self.data_path = data_path
        self.split_path = split_path
        self.image_size = image_size

        self.return_depth = return_depth
        self.full_size_depth = full_size_depth
        self.return_flow = return_flow
        self.preprocessed_path = preprocessed_path

        self.frame_count = frame_count
        self.keyframe_offset = keyframe_offset
        self.dilation = dilation

        self.selected_sequences = selected_sequences

        self._left_offset = (
            (self.frame_count - 1) // 2 + self.keyframe_offset
        ) * self.dilation

        if self.split_path is not None:
            self._datapoints = self._load_split(self.split_path)
        else:
            self._datapoints = None

        self._sequences, self._seq_data = self._get_sequences(self.data_path)

        if self._datapoints is None:
            self._datapoints = self._full_split(self._sequences, self._left_offset, (self.frame_count - 1) * dilation, sequence_sampler)

        if flow_selector is None:
            self.flow_selector = get_flow_selector(self.frame_count)
        else:
            self.flow_selector = flow_selector
        
        self.index_selector = index_selector

        self.length = len(self._datapoints)

    def _get_sequences(self, data_path: str):
        sequences = {}
        seq_data = {}

        logger.propagate = False

        seqs = []

        for seq in tqdm(os.listdir(os.path.join(data_path))):
            if os.path.isdir(os.path.join(data_path, seq)):

                seqs.append(seq)

        seqs = list(sorted(seqs))

        if self.selected_sequences is not None:
            seqs = [seqs[i] for i in self.selected_sequences]

        if self._datapoints is not None:
            relevant_seqs = set([seq for seq, _ in self._datapoints])
            seqs = [seq for seq in seqs if seq in relevant_seqs]

        def load_seq(seq):
            seq_path = os.path.join(data_path, seq)

            # print("Aea data provider", seq)
            aea_data_provider = AriaEverydayActivitiesDataProvider(seq_path)

            # print("MPS data provider")
            mps_data_paths_provider = mps.MpsDataPathsProvider(seq_path + "/mps")
            mps_data_provider = mps.MpsDataProvider(mps_data_paths_provider.get_data_paths())

            device_calibration = aea_data_provider.vrs.get_device_calibration()

            # print("Device times")
            device_times = aea_data_provider.vrs.get_timestamps_ns(RGB_STREAM_ID, TimeDomain.DEVICE_TIME)

            rgb_calib = device_calibration.get_camera_calib(aea_data_provider.vrs.get_label_from_stream_id(RGB_STREAM_ID))

            focal_lengths = rgb_calib.get_focal_lengths()
            image_size = rgb_calib.get_image_size()
            pinhole_calib = calibration.get_linear_camera_calibration(
                image_size[0], image_size[1], focal_lengths[0]
            )

            _seq_data = {
                "aea_data_provider": aea_data_provider,
                "mps_data_provider": mps_data_provider,
                "device_times": device_times,
                "fisheye_calib": rgb_calib,
                "pinhole_calib": pinhole_calib,
            }

            seq_len = len(device_times)

            return (seq, seq_len, _seq_data)

        pool = ThreadPool(8)
        for seq, seq_len, _seq_data in [load_seq(seq) for seq in tqdm(seqs)]:
            sequences[seq] = seq_len
            seq_data[seq] = _seq_data

        logger.propagate = True
        
        return sequences, seq_data

    @staticmethod
    def _full_split(sequences: dict, left_offset: int = 0, sub_seq_len: int = 2, sequence_sampler=None):
        datapoints = []
        for seq, seq_len in sequences.items():
            if sequence_sampler is not None:
                datapoints.extend(sequence_sampler(seq, seq_len, left_offset, sub_seq_len))
            else:
                if seq_len < sub_seq_len:
                    continue
                for i in range(seq_len - 1): # -1 because we need at least two frames
                    datapoints.append((seq, i))
        return datapoints

    @staticmethod
    def _load_split(split_path: str):
        with open(split_path, "r") as f:
            lines = f.readlines()

        def split_line(l):
            segments = l.split(" ")
            seq = segments[0]
            id = int(segments[1])
            return seq, id

        return list(map(split_line, lines))

    def __len__(self):
        return len(self._datapoints)
    
    def load_images(self, seq: str, ids: list):
        imgs = []

        for id in ids:
            t = self._seq_data[seq]["device_times"][id]

            img = self._seq_data[seq]["aea_data_provider"].vrs.get_image_data_by_time_ns(RGB_STREAM_ID, t, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST)
            img = img[0].to_numpy_array()

            img = distort_by_calibration(img, self._seq_data[seq]["pinhole_calib"], self._seq_data[seq]["fisheye_calib"])

            img = np.flip(img.transpose(1, 0, 2), axis=1)

            img = img.astype(np.float32) / 255.0

            imgs += [img]

        return imgs

    def get_poses(self, seq, ids):
        poses = []

        mps_data_provider = self._seq_data[seq]["mps_data_provider"]
        cam2device = transform3x4to4x4(self._seq_data[seq]["fisheye_calib"].get_transform_device_camera().to_matrix3x4())

        for id in ids:
            t = self._seq_data[seq]["device_times"][id]
            device2world = transform3x4to4x4(mps_data_provider.get_closed_loop_pose(t, TimeQueryOptions.CLOSEST).transform_world_device.to_matrix3x4())

            pose = device2world @ cam2device @ XY_FLIP

            poses += [pose]

        return poses
    
    def get_projs(self, seq, ids):
        projs = []

        calib = self._seq_data[seq]["pinhole_calib"]
        focal_lengths = calib.get_focal_lengths()
        principal_point = calib.get_principal_point()
        intrinsics = np.array([
            [focal_lengths[0], 0, principal_point[0]],
            [0, focal_lengths[1], principal_point[1]],
            [0, 0, 1],
        ])
        
        for id in ids:
            projs += [intrinsics]

        return projs

    def load_depth(self, seq, id, target_size=None, crop=None):
        raise NotImplementedError()

    def load_flows(self, seq, ids):
        raise NotImplementedError()

    def _index_to_seq_ids(self, index):
        if index >= self.length:
            raise IndexError()

        sequence, id = self._datapoints[index]
        seq_len = self._sequences[sequence]

        if self.index_selector is not None:
            ids = self.index_selector(id, self.frame_count, self.dilation, self._left_offset)
        else:
            ids = [id] + [i
                for i in range(
                    id - self._left_offset,
                    id - self._left_offset + self.frame_count * self.dilation,
                    self.dilation,
                )
                if i != id
            ]

        ids = [max(min(i, seq_len - 1), 0) for i in ids]

        return sequence, ids

    def __getitem__(self, index):
        sequence, ids = self._index_to_seq_ids(index)

        imgs = self.load_images(sequence, ids)

        original_size = imgs[0].shape[:2]

        # This will make sure that there are no invalid pixels in the corners
        target_size, crop = get_target_size_and_crop_custom(self.image_size, original_size, invalid_sides=0.045)

        if self.return_depth:
            depth = self.load_depth(sequence, ids[0], target_size, crop)
        else:
            depth = None
        
        if self.return_flow:
            flows_fwd, flows_bwd = self.load_flows(sequence, ids)
        else:
            flows_fwd = None
            flows_bwd = None

        imgs = np.stack([process_img(img, target_size, crop) for img in imgs])

        if self.return_flow:
            flows_fwd = np.stack([process_flow(flow, target_size, crop) for flow in flows_fwd])
            flows_bwd = np.stack([process_flow(flow, target_size, crop) for flow in flows_bwd])

        projs = self.get_projs(sequence, ids)
        projs = [process_proj(proj, original_size, target_size, crop) for proj in projs]

        projs = np.stack(projs)

        poses = np.stack(self.get_poses(sequence, ids))

        data = {
            "imgs": imgs,
            "projs": projs,
            "poses": poses,
            "ids": np.array(ids, dtype=np.int64),
            "data_id": index,
        }

        if self.return_depth:
            data["depths"] = depth[None, ...]
        
        if self.return_flow:
            data["flows_fwd"] = flows_fwd
            data["flows_bwd"] = flows_bwd

        return data

    def get_img_paths(self, index):
        raise NotImplementedError()

    def get_sequence(self, index: int):
        sequence, _ = self._index_to_seq_ids(index)
        return sequence


class ExtractedAEADataset(AriaEADataset):
    def _get_sequences(self, data_path: str):
        sequences = {}
        seq_data = {}

        seqs = []

        for seq in tqdm(os.listdir(os.path.join(data_path))):
            if os.path.isdir(os.path.join(data_path, seq)):
                seqs.append(seq)

        seqs = list(sorted(seqs))

        sequences = {
            seq: len(os.listdir(os.path.join(data_path, seq, "frames"))) for seq in seqs
        }

        for seq in seqs:
            poses = np.load(os.path.join(data_path, seq, "poses.npy")).astype(np.float32)
            projs = np.load(os.path.join(data_path, seq, "projs.npy")).astype(np.float32)

            seq_data[seq] = {
                "poses": poses,
                "projs": projs,
            }

        return sequences, seq_data
    
    def load_images(self, seq: str, ids: list):
        imgs = []

        for id in ids:
            img = cv2.cvtColor(cv2.imread(os.path.join(self.data_path, seq, "frames", f"{id:05d}.png")), cv2.COLOR_BGR2RGB)

            img = img.astype(np.float32) / 255.0

            imgs += [img]

        return imgs

    def get_poses(self, seq, ids):
        poses = []

        _poses = self._seq_data[seq]["poses"]

        for id in ids:
            poses += [_poses[id]]

        return poses
    
    def get_projs(self, seq, ids):
        projs = []

        _projs = self._seq_data[seq]["projs"]

        for id in ids:
            projs += [_projs[id]]

        return projs