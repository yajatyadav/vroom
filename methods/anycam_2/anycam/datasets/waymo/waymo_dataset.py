import os
import pickle
import time
from pathlib import Path
from typing import Iterable, Optional
from dotdict import dotdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter

from anycam.common.augmentation import get_color_aug_fn
from anycam.common.cameras.pinhole import unnormalize_camera_intrinsics
from anycam.common.geometry import estimate_frustum_overlap_2
from anycam.common.io.io import load_flow
from anycam.common.point_sampling import regular_grid
from anycam.datasets.common import get_flow_selector, get_target_size_and_crop, process_flow, process_img, process_proj


class WaymoDataset(Dataset):
    NAME = "Waymo"

    def __init__(
        self,
        data_path: str,
        split_path: Optional[str],
        image_size: Optional[tuple] = (320, 480),
        frame_count: int = 2,
        keyframe_offset: int = 0,
        dilation: int = 1,
        return_depth: bool = False,
        full_size_depth: bool = False,
        return_flow: bool = False,
        preprocessed_path: Optional[str] = None,
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

        self._sequences = self._get_sequences(self.data_path)

        self._calibs = self._load_calibs(self.data_path, self._sequences)
        self._poses = self._load_poses(self.data_path, self._sequences)

        self._left_offset = (
            (self.frame_count - 1) // 2 + self.keyframe_offset
        ) * self.dilation

        if self.split_path is not None:
            self._datapoints = self._load_split(self.split_path)
        else:
            self._datapoints = self._full_split(self._sequences, self._poses, self._left_offset, (self.frame_count - 1) * self.dilation, sequence_sampler)

        if flow_selector is None:
            self.flow_selector = get_flow_selector(self.frame_count)
        else:
            self.flow_selector = flow_selector

        self.index_selector = index_selector

        self.length = len(self._datapoints)

    @staticmethod
    def _get_sequences(data_path: str):
        all_sequences = []

        seqs_path = Path(data_path)
        for seq in seqs_path.iterdir():
            if not seq.is_dir():
                continue
            all_sequences.append(seq.name)

        return all_sequences

    @staticmethod
    def _full_split(sequences: list, poses: dict, left_offset: int = 0, sub_seq_len: int = 2, sequence_sampler=None):
        datapoints = []
        for seq in sorted(sequences):
            seq_len = len(poses[seq])

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

    @staticmethod
    def _load_calibs(data_path: str, sequences: list):
        data_path = Path(data_path)

        calibs = {}

        for seq in sequences:
            seq_folder = data_path / seq

            with (seq_folder / "calibration.pkl").open(mode="rb") as f:
                seq_calib = pickle.load(f)

            seq_calib["proj_mats"] = {
                cam: np.array(unnormalize_camera_intrinsics(
                    seq_calib["proj_mats"][cam], seq_calib["dims"]
                ))
                for cam in seq_calib["proj_mats"]
            }

            calibs[seq] = seq_calib

        return calibs

    @staticmethod
    def _load_poses(data_path: str, sequences: list):
        poses = {}

        for seq in sequences:
            pose_file = Path(data_path) / seq / f"poses.npy"
            seq_poses = np.load(str(pose_file))

            poses[seq] = seq_poses

        return poses
    
    def __len__(self) -> int:
        return self.length

    def load_images(self, seq: str, ids: list):
        imgs = []

        for cam, id in ids:
            cam_name = f"cam_0{cam}"

            img = (
                cv2.cvtColor(
                    cv2.imread(
                        os.path.join(
                            self.data_path, seq, "frames", cam_name, f"{id:010d}.jpg"
                        )
                    ),
                    cv2.COLOR_BGR2RGB,
                ).astype(np.float32)
                / 255
            )
            imgs += [img]

        return imgs
    
    def load_flows(self, seq: str, ids: list):
        flows_fwd = []
        flows_bwd = []

        flow_ids = self.flow_selector([id[1] for id in ids])
        cam = ids[0][0]

        for (fwd_id, fwd_is_fwd), (bwd_id, bwd_is_fwd) in zip(*flow_ids):
            # print(fwd_id, fwd_is_fwd, bwd_id, bwd_is_fwd)
            cam_name = f"cam_0{cam}"
            flow_fwd = load_flow(
                        os.path.join(
                            self.preprocessed_path, seq, "unimatch_flows", cam_name, f"{fwd_id:010d}_{'fwd' if fwd_is_fwd else 'bwd'}.png"
                        )
                    )
            flow_bwd = load_flow(
                        os.path.join(
                            self.preprocessed_path, seq, "unimatch_flows", cam_name, f"{bwd_id:010d}_{'fwd' if bwd_is_fwd else 'bwd'}.png"
                        )
                    )
            
            flows_fwd += [flow_fwd]
            flows_bwd += [flow_bwd]

        return flows_fwd, flows_bwd

    def load_depth(self, seq, id, original_size, target_size=None, crop=None):
        if target_size is None and self.image_size is not None:
            depth_size = self.image_size
        elif target_size is not None:
            depth_size = target_size
        else:
            depth_size = original_size
        
        if self.full_size_depth:
            depth_size = (1280, 1920)
            crop = None

        cam, idx = id

        points = (
            np.load(os.path.join(self.data_path, seq, "lidar", f"{idx:010d}.npy"))
            .astype(dtype=np.float32)
            .reshape(-1, 3)
        )

        proj = self._calibs[seq]["proj_mats"][cam]

        proj = process_proj(proj, original_size, target_size, crop)

        points_hom = np.concatenate((points, np.ones_like(points[:, :1])), axis=1)
        points_cam = (
            (
                proj
                @ np.linalg.inv(self._calibs[seq]["extrinsics"][cam])[:3, :]
            )
            @ points_hom.T
        ).T
        points_cam[:, :2] = points_cam[:, :2] / np.clip(points_cam[:, 2:3], a_min=1e-4, a_max=None)

        if crop is None:
            h_out, w_out = depth_size
        else:
            h_out, w_out = crop[2:]

        mask = (
            (points_cam[:, 0] >= 0)
            & (points_cam[:, 0] < w_out)
            & (points_cam[:, 1] >= 0)
            & (points_cam[:, 1] < h_out)
            & (points_cam[:, 2] > 0)
        )
        points_cam = points_cam[mask, :]

        # project to image
        depth = np.zeros((h_out, w_out), dtype=np.float32)
        depth[
            points_cam[:, 1].astype(np.int32).clip(0, h_out - 1),
            points_cam[:, 0].astype(np.int32).clip(0, w_out - 1),
        ] = points_cam[:, 2]

        depth[depth < 0] = 0

        return depth[None, :, :]
    
    def _index_to_seq_ids(self, index):
        if index >= self.length:
            raise IndexError()

        cam = 1

        sequence, id = self._datapoints[index]
        seq_len = self._poses[sequence].shape[0]

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

        ids = [(cam, i) for i in ids]

        return sequence, ids

    def __getitem__(self, index: int):
        _start_time = time.time()

        sequence, ids = self._index_to_seq_ids(index)

        target_size, crop = get_target_size_and_crop(self.image_size, self._calibs[sequence]["dims"])     

        _start_time_loading = time.time()

        imgs = self.load_images(sequence, ids)
  
        if self.return_depth:
            depth = self.load_depth(sequence, ids[0], self._calibs[sequence]["dims"], target_size, crop)
        else:
            depth = None
        
        if self.return_flow:
            flows_fwd, flows_bwd = self.load_flows(sequence, ids)
        else:
            flows_fwd = None
            flows_bwd = None

        
        _loading_time = np.array(time.time() - _start_time_loading)

        _start_time_processing = time.time()

        imgs = np.stack([process_img(img, target_size, crop) for img in imgs])

        if self.return_flow:
            flows_fwd = np.stack([process_flow(flow, target_size, crop) for flow in flows_fwd])
            flows_bwd = np.stack([process_flow(flow, target_size, crop) for flow in flows_bwd])

        _processing_time = np.array(time.time() - _start_time_processing)

        # These poses are camera to world !!
        poses = np.stack([
            self._poses[sequence][i, :, :] @ self._calibs[sequence]["extrinsics"][cam]
            for cam, i in ids
        ])

        projs = np.stack([process_proj(self._calibs[sequence]["proj_mats"][cam], self._calibs[sequence]["dims"], target_size, crop) for cam, _ in ids])
        
        ids = np.array([id for cam, id in ids], dtype=np.int64)

        _proc_time = np.array(time.time() - _start_time)

        data = {
            "imgs": imgs,
            "projs": projs,
            "poses": poses,
            "ids": ids,
            "data_id": index,
            # "t__get_item__": np.array([_proc_time]),
        }

        if self.return_depth:
            data["depths"] = depth[None, ...]
        
        if self.return_flow:
            data["flows_fwd"] = flows_fwd
            data["flows_bwd"] = flows_bwd

        return data

    def get_img_paths(self, index: int):
        sequence, ids = self._index_to_seq_ids(index)
        img_paths = [
            os.path.join(self.data_path, sequence, "frames", f"cam_0{cam}", f"{id:010d}.jpg")
            for cam, id in ids
        ]

        return img_paths

    def get_sequence(self, index: int):
        sequence, _ = self._index_to_seq_ids(index)
        return sequence