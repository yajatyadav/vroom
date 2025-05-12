# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utility functions for global alignment
# --------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import zscore

def edge_str(i, j):
    return f'{i}_{j}'


def i_j_ij(ij):
    # inputs are (i, j)
    return edge_str(*ij), ij


def edge_conf(conf_i, conf_j, edge):

    score = float(conf_i[edge].mean() * conf_j[edge].mean())

    return score


def compute_edge_scores(edges, conf_i, conf_j):
    score_dict = {(i, j): edge_conf(conf_i, conf_j, e) for e, (i, j) in edges}

    return score_dict

def NoGradParamDict(x):
    assert isinstance(x, dict)
    return nn.ParameterDict(x).requires_grad_(False)


def get_imshapes(edges, pred_i, pred_j):
    n_imgs = max(max(e) for e in edges) + 1
    imshapes = [None] * n_imgs
    for e, (i, j) in enumerate(edges):
        shape_i = tuple(pred_i[e].shape[0:2])
        shape_j = tuple(pred_j[e].shape[0:2])
        if imshapes[i]:
            assert imshapes[i] == shape_i, f'incorrect shape for image {i}'
        if imshapes[j]:
            assert imshapes[j] == shape_j, f'incorrect shape for image {j}'
        imshapes[i] = shape_i
        imshapes[j] = shape_j
    return imshapes


def get_conf_trf(mode):
    if mode == 'log':
        def conf_trf(x): return x.log()
    elif mode == 'sqrt':
        def conf_trf(x): return x.sqrt()
    elif mode == 'm1':
        def conf_trf(x): return x-1
    elif mode in ('id', 'none'):
        def conf_trf(x): return x
    else:
        raise ValueError(f'bad mode for {mode=}')
    return conf_trf


def l2_dist(a, b, weight):
    return ((a - b).square().sum(dim=-1) * weight)


def l1_dist(a, b, weight):
    return ((a - b).norm(dim=-1) * weight)


ALL_DISTS = dict(l1=l1_dist, l2=l2_dist)


def signed_log1p(x):
    sign = torch.sign(x)
    return sign * torch.log1p(torch.abs(x))


def signed_expm1(x):
    sign = torch.sign(x)
    return sign * torch.expm1(torch.abs(x))


def cosine_schedule(t, lr_start, lr_end):
    assert 0 <= t <= 1
    return lr_end + (lr_start - lr_end) * (1+np.cos(t * np.pi))/2


def linear_schedule(t, lr_start, lr_end):
    assert 0 <= t <= 1
    return lr_start + (lr_end - lr_start) * t

def cycled_linear_schedule(t, lr_start, lr_end, num_cycles=2):
    assert 0 <= t <= 1
    cycle_t = t * num_cycles
    cycle_t = cycle_t - int(cycle_t)
    if t == 1:
        cycle_t = 1
    return linear_schedule(cycle_t, lr_start, lr_end)

# --------------------------------------------------------
# key-frame subsampling
# --------------------------------------------------------
import math
from typing import Sequence, Tuple, List, Union

_Tensor = Union[torch.Tensor, np.ndarray]


def _rotation_angle_deg(R: _Tensor) -> float:
    """
    Convert a 3×3 rotation matrix to the equivalent rotation angle in degrees.
    Uses the stable acos formulation:  θ = acos( (tr(R) − 1) / 2 ).
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    cos_theta = max(min((trace - 1.0) * 0.5, 1.0), -1.0)  # clamp for safety
    return math.degrees(math.acos(cos_theta))


def _pose_components(pose: _Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract translation (3,) and rotation matrix (3,3) from a pose representation.

    Supports:
        • 4×4 or 3×4 homogeneous matrix (torch or numpy)
        • 7-vector  [qw qx qy qz tx ty tz]   or   [qx qy qz qw tx ty tz]
    """
    if pose.shape == (4, 4) or pose.shape == (3, 4):
        R = pose[:3, :3]
        t = pose[:3, 3]
    elif pose.shape[-1] == 7:
        # Assume (q, t) where q = (w, x, y, z) or (x, y, z, w)
        q = np.asarray(pose[:4], dtype=float)
        if abs(np.linalg.norm(q) - 1.0) > 1e-3:        # not normalised?
            q = q / np.linalg.norm(q)
        # map to w, x, y, z ordering
        if q[0] < -0.5 or q[0] > 0.5:                  # heuristic: probably w first
            w, x, y, z = q
        else:                                          # x, y, z, w
            x, y, z, w = q
        # quaternion to R
        R = np.array([
            [1 - 2 * (y**2 + z**2),     2 * (x * y - z * w),     2 * (x * z + y * w)],
            [    2 * (x * y + z * w), 1 - 2 * (x**2 + z**2),     2 * (y * z - x * w)],
            [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
        ], dtype=float)
        t = np.asarray(pose[4:], dtype=float)
    else:
        raise ValueError(f'Unsupported pose shape {pose.shape}')

    return np.asarray(t, dtype=float), np.asarray(R, dtype=float)


def select_keyframes(
    poses: Sequence[_Tensor],
    trans_thresh: float = 0.50,
    rot_thresh_deg: float = 3.0,
    feat_alive: Sequence[float] | None = None,
    feat_thresh: float = 0.30,
    always_keep_ends: bool = True,
) -> Tuple[List[bool], List[int]]:
    """
    Decide which frames to keep as key-frames.

    Args
    ----
    poses
        Iterable of camera poses in any of the supported formats.
    trans_thresh
        Minimum translation (metres) since the last key-frame to trigger a new one.
    rot_thresh_deg
        Minimum rotation (degrees) since the last key-frame to trigger a new one.
    feat_alive
        Optional list/array with the fraction of surviving feature tracks per frame.
        If provided, we also declare a key-frame when `feat_alive[i] < feat_thresh`.
    feat_thresh
        Threshold used with `feat_alive`.
    always_keep_ends
        If True (default) the first and last frames are forced to be key-frames.

    Returns
    -------
    mask
        Boolean list of length N (True = key-frame).
    kf_indices
        Sorted list of key-frame indices (a convenience duplicate of `np.flatnonzero(mask)`).
    """
    n = len(poses)
    if n == 0:
        return [], []

    mask = [False] * n
    last_kf = 0
    mask[0] = True  # always keep first

    t_last, R_last = _pose_components(poses[0])

    for i in range(1, n):
        t_i, R_i = _pose_components(poses[i])
        d_t = float(np.linalg.norm(t_i - t_last))
        d_r = float(_rotation_angle_deg(R_last.T @ R_i))

        feat_cond = (feat_alive is not None) and (feat_alive[i] < feat_thresh)
        if d_t > trans_thresh or d_r > rot_thresh_deg or feat_cond:
            mask[i] = True
            last_kf = i
            t_last, R_last = t_i, R_i

    if always_keep_ends and not mask[-1]:
        mask[-1] = True

    kf_indices = [i for i, m in enumerate(mask) if m]
    return mask, kf_indices
