import os
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets._optical_flow import _read_flo

from anycam.common.io.io import load_array, load_flow
from anycam.datasets.common import get_flow_selector, get_target_size_and_crop, process_depth, process_flow, process_img, process_proj


_EPS = np.finfo(float).eps * 4.0


def transform44(l):
    """
    Generate a 4x4 homogeneous transformation matrix from a 3D point and unit quaternion.
    
    Input:
    l -- tuple consisting of (tx,ty,tz,qx,qy,qz,qw) where
         (tx,ty,tz) is the 3D position and (qx,qy,qz,qw) is the unit quaternion.
         
    Output:
    matrix -- 4x4 homogeneous transformation matrix
    """
    t = l[0:3]
    q = np.array(l[3:7], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.array((
        (                1.0,                 0.0,                 0.0, t[0]),
        (                0.0,                 1.0,                 0.0, t[1]),
        (                0.0,                 0.0,                 1.0, t[2]),
        (                0.0,                 0.0,                 0.0, 1.0),
        ), dtype=np.float64)
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], t[0]),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], t[1]),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], t[2]),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)


axis_flipper = np.array([
    [1, 0, 0, 0],
    [0, 0, -1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
])


class TUMRGBDDataset(Dataset):
    NAME = "TUM-RGBD"

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

        self._sequences, self._seq_data = self._get_sequences(self.data_path)

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
        seq_data = {}

        for seq in os.listdir(os.path.join(data_path)):
            if os.path.isdir(os.path.join(data_path, seq)):
                with open(os.path.join(data_path, seq, "rgb.txt"), "r") as f:
                    rgb_ts = [l for l in f.readlines() if not l.startswith("#")]
                with open(os.path.join(data_path, seq, "depth.txt"), "r") as f:
                    depth_ts = [l for l in f.readlines() if not l.startswith("#")]
                with open(os.path.join(data_path, seq, "groundtruth.txt"), "r") as f:
                    gt_ts = [l for l in f.readlines() if not l.startswith("#")]

                rgb_ts_np = np.array([float(l.split(" ")[0]) for l in rgb_ts])
                depth_ts_np = np.array([float(l.split(" ")[0]) for l in depth_ts])
                gt_ts_np = np.array([float(l.split(" ")[0]) for l in gt_ts])

                rgb_ts_np = np.stack([rgb_ts_np, np.arange(len(rgb_ts_np))], axis=1)
                depth_ts_np = np.stack([depth_ts_np, np.arange(len(depth_ts_np))], axis=1)
                gt_ts_np = np.stack([gt_ts_np, np.arange(len(gt_ts_np))], axis=1)

                T_0 = transform44([float(p) for p in gt_ts[0].split(" ")[1:]])

                min_t = max(rgb_ts_np[0, 0], depth_ts_np[0, 0], gt_ts_np[0, 0])
                max_t = min(rgb_ts_np[-1, 0], depth_ts_np[-1, 0], gt_ts_np[-1, 0])

                gt_ts_np = gt_ts_np[(gt_ts_np[:, 0] >= min_t) & (gt_ts_np[:, 0] <= max_t)]

                matches = []
                for i in range(len(gt_ts_np)):
                    gt_id = gt_ts_np[i, 1]
                    t = gt_ts_np[i, 0]

                    rgb_id = rgb_ts_np[np.argmin(np.abs(rgb_ts_np[:, 0] - t)), 1]
                    depth_id = depth_ts_np[np.argmin(np.abs(depth_ts_np[:, 0] - t)), 1]

                    pose_data = [float(p) for p in gt_ts[int(gt_id)].split(" ")[1:]]

                    pose = transform44(pose_data)

                    pose = axis_flipper @ pose
                    
                    gt_t = gt_ts[int(gt_id)].split(" ")[0].strip()
                    rgb_t = rgb_ts[int(rgb_id)].split(" ")[1].strip()
                    depth_t = depth_ts[int(depth_id)].split(" ")[1].strip()

                    matches.append((pose, rgb_t, depth_t, gt_t))

                seq_len = len(matches)
                sequences[seq] = seq_len
                seq_data[seq] = matches
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
            rgb_path = self._seq_data[seq][id][1]
            img = (
                cv2.cvtColor(
                    cv2.imread(
                        os.path.join(
                            self.data_path, seq, rgb_path
                        )
                    ),
                    cv2.COLOR_BGR2RGB,
                ).astype(np.float32)
                / 255
            )
            imgs += [img]

        return imgs

    def load_depth(self, seq, id, target_size=None, crop=None):
        depth_path = self._seq_data[seq][id][2]
        depth = load_array(os.path.join(self.data_path, seq, depth_path))
        depth = depth * (65535 / 5000)

        depth = process_depth(depth, target_size, crop)

        return depth

    def load_flows(self, seq, ids):
        raise NotImplementedError()

        flows_fwd = []
        flows_bwd = []

        flow_ids = self.flow_selector(ids)

        for (fwd_id, fwd_is_fwd), (bwd_id, bwd_is_fwd) in zip(*flow_ids):
            flow_fwd = load_flow(os.path.join(self.preprocessed_path, "unimatch_flows", seq, f"{fwd_id*5:05d}_{'fwd' if fwd_is_fwd else 'bwd'}.png"))
            flow_bwd = load_flow(os.path.join(self.preprocessed_path, "unimatch_flows", seq, f"{bwd_id*5:05d}_{'fwd' if bwd_is_fwd else 'bwd'}.png"))
        
            flows_fwd += [flow_fwd]
            flows_bwd += [flow_bwd]
        
        return flows_fwd, flows_bwd

    def get_proj(self, n):
        """ Read camera data, return (M,N) tuple.
        
        M is the intrinsic matrix, N is the extrinsic matrix, so that

        x = M*N*X,
        with x being a point in homogeneous image pixel coordinates, X being a
        point in homogeneous world coordinates.
        """

        proj = np.array([
            [525.0, 0, 319.5],
            [0, 525.0, 239.5],
            [0, 0, 1],
        ])

        proj = [proj] * n

        return proj
    
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

        # for id in ids:
        # print(self._seq_data[sequence][ids[0]][1:])

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

        projs = self.get_proj(len(ids))
        projs = [process_proj(proj, original_size, target_size, crop) for proj in projs]

        projs = np.stack(projs).astype(np.float32)

        poses = np.stack([np.array(self._seq_data[sequence][id][0]) for id in ids], axis=0)

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

        img_paths = [os.path.join(self.data_path, sequence, self._seq_data[sequence][id][1]) for id in ids]

        return img_paths

    def get_sequence(self, index: int):
        sequence, _ = self._index_to_seq_ids(index)
        return sequence
