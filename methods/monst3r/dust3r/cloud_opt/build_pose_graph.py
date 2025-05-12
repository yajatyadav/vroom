#!/usr/bin/env python3
"""
Build and *optimise* a (key-frame-reduced) pose-graph from MonST3R chunks.

Steps
   load every chunk produced by `get_reconstructed_scene`
   concatenate their poses / intrinsics
   pick key-frames with `select_keyframes`
   add odometry + loop-closure edges, optimise a global Sim(3) graph
   write JSON + corrected KF poses

Examples
--------
python -m dust3r.cloud_opt.build_pose_graph \
       --chunks runs/f1/chunk_* \
       --trans-thr 0.5 --rot-thr 3
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import imageio.v3 as iio
from scipy.spatial.transform import Rotation as SciRot

# ---------- MonST3R helpers --------------------------------------------------
from dust3r.cloud_opt.commons import select_keyframes
from dust3r.cloud_opt.pose_graph import PoseGraph
from dust3r.cloud_opt.loop_search import build_loop_edges
from dust3r.utils.device import to_numpy
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------#
# Utilities                                                                    #
# -----------------------------------------------------------------------------#
def tum_to_c2w_matrix(tum_row: np.ndarray) -> torch.Tensor:
    """[x y z  qw qx qy qz] → 4×4 cam-to-world matrix (torch.float32)."""
    xyz   = tum_row[:3]
    qwxyz = tum_row[3:]
    R     = SciRot.from_quat([qwxyz[1], qwxyz[2], qwxyz[3], qwxyz[0]]).as_matrix()
    M     = torch.eye(4, dtype=torch.float32)
    M[:3, :3] = torch.from_numpy(R).float()
    M[:3, 3]  = torch.from_numpy(xyz).float()
    return M


def load_chunk(chunk_dir: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns
    -------
    poses_c2w : (N,4,4) torch.float32
    K          : (N,3,3) torch.float32
    """
    traj = np.loadtxt(chunk_dir / "pred_traj.txt", dtype=np.float32)
    intr = np.loadtxt(chunk_dir / "pred_intrinsics.txt", dtype=np.float32)
    poses = torch.stack([tum_to_c2w_matrix(r) for r in traj], 0)
    K     = torch.tensor(intr.reshape(-1, 3, 3), dtype=torch.float32)
    return poses, K


def expand_chunk_glob(patterns: List[str]) -> List[Path]:
    out: List[Path] = []
    for pat in patterns:
        if any(ch in pat for ch in "*?[]"):
            out.extend(sorted(Path().glob(pat)))
        else:
            out.append(Path(pat))
    return out


def load_rgb(frame_path: Path) -> np.ndarray:
    """PNG uint8 HxWx3  (drops alpha if present)."""
    im = iio.imread(frame_path)
    if im.shape[-1] == 4:
        im = im[..., :3]
    return im


# -----------------------------------------------------------------------------#
# CLI                                                                          #
# -----------------------------------------------------------------------------#
def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("MonST3R pose-graph builder")
    p.add_argument("--chunks", nargs="+", required=True,
                   help="List / glob of chunk directories")
    p.add_argument("--trans-thr", type=float, default=0.50,
                   help="Key-frame translation threshold (m)")
    p.add_argument("--rot-thr", type=float, default=3.0,
                   help="Key-frame rotation threshold (deg)")
    p.add_argument("--feat-file", default="",
                   help="Optional .npy with per-frame surviving-feature ratio")
    p.add_argument("--fps", type=float, default=30,
                   help="Video frame-rate for loop-closure temporal filter")
    p.add_argument("--out", type=Path, default="kf_selection.json",
                   help="Output JSON (a .npy with optimised poses is also written)")
    return p


# -----------------------------------------------------------------------------#
# Main                                                                         #
# -----------------------------------------------------------------------------#
def main() -> None:
    args = make_argparser().parse_args()
    # import ipdb; ipdb.set_trace()
    # ❶ Load all chunks --------------------------------------------------------
    chunk_dirs = expand_chunk_glob(args.chunks)
    if not chunk_dirs:
        raise ValueError("No chunk directories matched --chunks")

    all_poses, chunk_lengths = [], []
    for cdir in chunk_dirs:
        poses, _ = load_chunk(cdir)
        all_poses.append(poses)
        chunk_lengths.append(len(poses))

    poses_cat = torch.cat(all_poses, 0)                        # (N,4,4)
    print(f"[build_pose_graph] {poses_cat.shape[0]} total frames "
          f"across {len(chunk_dirs)} chunks")

    # (optional) feature-alive curve ------------------------------------------
    feat_alive = None
    if args.feat_file:
        feat_alive = np.load(args.feat_file).tolist()
        assert len(feat_alive) == poses_cat.shape[0], \
            "--feat-file length mismatch with total frames"

    # ❷ Key-frame selection ----------------------------------------------------
    mask_bool, kf_ids = select_keyframes(
        poses=to_numpy(poses_cat),
        trans_thresh=args.trans_thr,
        rot_thresh_deg=args.rot_thr,
        feat_alive=feat_alive,
    )
    print(f"[build_pose_graph] kept {len(kf_ids)} / {poses_cat.shape[0]} "
          f"frames as key-frames ({100*len(kf_ids)/poses_cat.shape[0]:.1f} %)")

    # ➍ Pose-graph: odometry + loop closures + optimisation --------------------
    device = "cpu"      # graph is light – keep on CPU
    pg = PoseGraph(len(kf_ids), device=device)
    pg.T[:] = poses_cat[kf_ids].to(device)

    #  ➍-a  odometry edges inside each chunk
    off = 0
    for L in chunk_lengths:
        # global indices of KFs that fall inside this chunk
        kfs_in_chunk = [idx for idx in kf_ids if off <= idx < off + L]
        if len(kfs_in_chunk) > 1:
            local_pose_subset = poses_cat[kfs_in_chunk]
            pg.add_odometry(kfs_in_chunk, local_pose_subset)
        off += L

    #  ➍-b  loop-closure edges (optional if FAISS / SuperGlue are absent)
    print("[build_pose_graph] searching loop closures …")
    # gather RGB frames of *key-frames only*  (slowest step: PNG I/O)
    rgb_kf: List[np.ndarray] = []
    off = 0
    for cdir, L in zip(chunk_dirs, chunk_lengths):
        for local_idx in range(L):
            g_idx = off + local_idx
            if g_idx in kf_ids:
                rgb_kf.append(load_rgb(cdir / f"frame_{local_idx:04d}.png"))
        off += L

    loop_edges = build_loop_edges(rgb_kf,
                                  poses_cat[kf_ids].cpu().numpy(),
                                  fps=args.fps,
                                  k=20,
                                  min_dt=15.0,
                                  device="cuda" if torch.cuda.is_available() else "cpu")
    for i, j, Tij, w in loop_edges:
        pg.add_edge(i, j, Tij.to(device), torch.eye(6, device=device) * w)

    #  ➍-c  optimise Sim(3)  (pyg2o if available, else no-op)
    pg.optimise(iters=100, fix_first=True)

    # ➎ Persist ----------------------------------------------------------------
    out_json = args.out
    out_npy  = out_json.with_suffix(".poses.npy")
    record = dict(
        chunk_dirs     =[str(p) for p in chunk_dirs],
        chunk_lengths  =chunk_lengths,
        kf_mask        =mask_bool,
        kf_indices     =kf_ids,
        trans_thr      =args.trans_thr,
        rot_thr        =args.rot_thr,
        feat_file      =args.feat_file or None,
        corrected_pose =str(out_npy),
    )
    out_json.write_text(json.dumps(record, indent=2))
    np.save(out_npy, pg.T.cpu().numpy())
    print(f"[build_pose_graph] wrote key-frame spec → {out_json}")
    print(f"[build_pose_graph] wrote optimised KF poses → {out_npy}")


if __name__ == "__main__":
    main()
