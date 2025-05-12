import os
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets._optical_flow import _read_flo

from anycam.common.io.io import load_flow
from anycam.datasets.common import get_flow_selector, get_target_size_and_crop, process_depth, process_flow, process_img, process_proj


class YouTubeVOSDataset(Dataset):
    NAME = "YouTube-VOS"

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

        self._left_offset = (
            (self.frame_count - 1) // 2 + self.keyframe_offset
        ) * self.dilation

        self._sequences, self._id_to_file_id = self._get_sequences(self.data_path)

        if self.split_path is not None:
            self._datapoints = self._load_split(self.split_path)
        else:
            self._datapoints = self._full_split(self._sequences, self._left_offset, (self.frame_count - 1) * dilation, sequence_sampler)

        if flow_selector is None:
            self.flow_selector = get_flow_selector(self.frame_count)
        else:
            self.flow_selector = flow_selector

        self.index_selector = index_selector

        self.length = len(self._datapoints)

    @staticmethod
    def _get_sequences(data_path: str):
        sequences = {}
        id_to_file_id = {}
        for seq in os.listdir(os.path.join(data_path, "JPEGImages")):
            if os.path.isdir(os.path.join(data_path, "JPEGImages", seq)):
                img_list = list(sorted(os.listdir(os.path.join(data_path, "JPEGImages", seq))))
                img_ids = [img.split(".")[0] for img in img_list]
                seq_len = len(img_list)
                sequences[seq] = seq_len
                id_to_file_id[seq] = img_ids
        return sequences, id_to_file_id

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
            img = (
                cv2.cvtColor(
                    cv2.imread(
                        os.path.join(
                            self.data_path, "JPEGImages", seq, f"{self._id_to_file_id[seq][id]}.jpg"
                        )
                    ),
                    cv2.COLOR_BGR2RGB,
                ).astype(np.float32)
                / 255
            )
            imgs += [img]

        return imgs

    def load_flows(self, seq, ids):
        flows_fwd = []
        flows_bwd = []


        flow_ids = self.flow_selector(ids)

        for (fwd_id, fwd_is_fwd), (bwd_id, bwd_is_fwd) in zip(*flow_ids):
            flow_fwd = load_flow(os.path.join(self.preprocessed_path, "unimatch_flows", seq, f"{self._id_to_file_id[seq][fwd_id]}_{'fwd' if fwd_is_fwd else 'bwd'}.png"))
            flow_bwd = load_flow(os.path.join(self.preprocessed_path, "unimatch_flows", seq, f"{self._id_to_file_id[seq][bwd_id]}_{'fwd' if bwd_is_fwd else 'bwd'}.png"))
        
            flows_fwd += [flow_fwd]
            flows_bwd += [flow_bwd]
        
        return flows_fwd, flows_bwd

    def load_depth(self, seq, id, target_size=None, crop=None):
        """ Read depth data from file, return as numpy array. """

        assert target_size is not None

        depth = np.ones(dtype=np.float32, shape=(1, *target_size))

        return depth


    def get_dummy_cams(self, target_size, n):
        """ Read camera data, return (M,N) tuple.
        
        M is the intrinsic matrix, N is the extrinsic matrix, so that

        x = M*N*X,
        with x being a point in homogeneous image pixel coordinates, X being a
        point in homogeneous world coordinates.
        """
        if target_size is None:
            target_size = (256, 256)

        poses, projs = [], []

        for id in range(n):
            pose = np.eye(4, dtype=np.float32)
            proj = np.eye(3, dtype=np.float32)
            proj[0, 0] = target_size[0]
            proj[1, 1] = target_size[0]
            proj[0, 2] = target_size[0] / 2
            proj[1, 2] = target_size[1] / 2

            poses += [pose]
            projs += [proj]

        return projs, poses
    
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

        target_size, crop = get_target_size_and_crop(self.image_size, original_size)

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

        projs, poses = self.get_dummy_cams(target_size, len(ids))

        projs = np.stack(projs)
        poses = np.stack(poses)

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
        sequence, ids = self._index_to_seq_ids(index)

        img_paths = [
            os.path.join(self.data_path, "JPEGImages", sequence, f"{self._id_to_file_id[sequence][id]}.jpg")
            for id in ids
        ]

        return img_paths

    def get_sequence(self, index: int):
        sequence, _ = self._index_to_seq_ids(index)
        return sequence
