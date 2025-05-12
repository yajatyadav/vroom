# --------------------------------------------------------
# Very-light Sim(3) pose-graph container for MonST3R
# --------------------------------------------------------
#
#   PoseGraph  →  add_edge(...) / add_odometry(...)
#                   ↓
#                optimise()          (uses pyg2o if available,
#                                      otherwise returns identities)
#                   ↓
#            self.T  (N,4,4) cam-to-world matrices
#
# All tensors live on `self.device`, default = 'cpu'.
# --------------------------------------------------------
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation as SciRot   #  ← add this

# ------------------------------------------------------------------ #
# Sim(3) helpers                                                     #
# ------------------------------------------------------------------ #
def sim3_mul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Matrix product for (…,4,4) Sim(3) tensors."""
    return A @ B


def se3_log(M: torch.Tensor) -> torch.Tensor:
    """
    Log-map of an SE(3) matrix → 6-vector  (ω, v).
    Implemented in PyTorch for small residuals ≈ used in optimisation.
    """
    R, t = M[:3, :3], M[:3, 3]
    th = torch.arccos(torch.clamp((torch.trace(R) - 1) / 2, -1 + 1e-6, 1 - 1e-6))
    if th.abs() < 1e-8:
        omega = torch.zeros(3, device=M.device)
        V_inv = torch.eye(3, device=M.device)
    else:
        omega = th / (2 * torch.sin(th)) * torch.tensor(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
            device=M.device,
        )
        A = torch.sin(th) / th
        B = (1 - torch.cos(th)) / (th * th)
        Wx = _skew(omega)
        V_inv = torch.eye(3, device=M.device) - 0.5 * Wx + (1 / th**2) * (
            1 - A / (2 * B)
        ) * Wx @ Wx
    v = V_inv @ t
    return torch.cat([omega, v])


def _skew(w: torch.Tensor) -> torch.Tensor:
    """3-vector → 3×3 skew matrix."""
    return torch.tensor(
        [[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]], device=w.device
    )


# ------------------------------------------------------------------ #
# PoseGraph class                                                    #
# ------------------------------------------------------------------ #
class PoseGraph:
    """
    Minimal Sim(3) pose-graph.

    All units: metres & radians.
    """

    def __init__(self, n_nodes: int, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.T = torch.eye(4, device=self.device).repeat(n_nodes, 1, 1)  # (N,4,4)
        self.edges: List[Tuple[int, int, torch.Tensor, torch.Tensor]] = []
        # Each edge:  (idx_i, idx_j, T_meas 4×4, info 6×6)

    # ------------------------------------------------------------------ #
    # edge helpers                                                       #
    # ------------------------------------------------------------------ #
    def add_edge(
        self,
        i: int,
        j: int,
        T_ij: torch.Tensor,
        info_6x6: torch.Tensor,
    ) -> None:
        self.edges.append(
            (int(i), int(j), T_ij.to(self.device), info_6x6.to(self.device))
        )

    def add_odometry(
        self,
        kf_indices: List[int],
        poses_w2c: torch.Tensor,
        sigma_t: float = 0.02,
        sigma_r: float = math.radians(1),
    ) -> None:
        """
        `kf_indices`  – indices *inside the global KF list* that correspond to
        consecutive frames of the same chunk;
        `poses_w2c`   – subset of self.T *before* optimisation ( (M,4,4) ).
        """
        info = torch.diag(
            torch.tensor(
                [1 / sigma_r**2] * 3 + [1 / sigma_t**2] * 3, device=self.device
            )
        )
        for a, b in zip(range(len(kf_indices) - 1), range(1, len(kf_indices))):
            i, j = kf_indices[a], kf_indices[b]
            Tij = poses_w2c[b] @ torch.linalg.inv(poses_w2c[a])
            self.add_edge(i, j, Tij, info)

    # ------------------------------------------------------------------ #
    # Optimisation                                                       #
    # ------------------------------------------------------------------ #
    def optimise(self, iters: int = 100, fix_first: bool = True) -> None:
        """
        Global LM over Sim(3).  Uses pyg2o if available, else no-op.
        """
        try:
            import g2o as pyg2o  # type: ignore
        except ImportError:
            print(
                "[pose_graph] pyg2o not found – skipping optimisation "
                "(trajectory will stay concatenated)."
            )
            return

        opt = pyg2o.SparseOptimizer()
        solver = pyg2o.BlockSolverSE3(
            pyg2o.LinearSolverCSparseSE3())  # SE3 (no scale) but OK
        alg = pyg2o.OptimizationAlgorithmLevenberg(solver)
        opt.set_algorithm(alg)

        # 1 – vertices
        for idx, T in enumerate(self.T):
            v = pyg2o.VertexSE3()
            v.set_id(idx)
            v.set_estimate(pyg2o.Isometry3d(T.cpu().numpy()))
            v.set_fixed(idx == 0 and fix_first)
            opt.add_vertex(v)

        # 2 – edges
        for i, j, T_ij, info in self.edges:
            e = pyg2o.EdgeSE3()
            e.set_vertex(0, opt.vertex(i))
            e.set_vertex(1, opt.vertex(j))
            e.set_measurement(pyg2o.Isometry3d(T_ij.cpu().numpy()))
            e.set_information(info.cpu().numpy())
            opt.add_edge(e)

        opt.initialize_optimization()
        print(f"[pose_graph] optimising {len(self.T)} nodes / {len(self.edges)} edges")
        opt.optimize(iters)

        # copy back to torch tensor
        for idx in range(len(self.T)):
            self.T[idx] = torch.from_numpy(
                opt.vertex(idx).estimate().matrix()).to(self.device)

    # ------------------------------------------------------------------ #
    # I/O convenience                                                    #
    # ------------------------------------------------------------------ #
    def save_txt(self, path: Path) -> None:
        """
        Write poses in TUM format  x y z qw qx qy qz
        """
        mat = self.T.cpu().numpy()
        xyz = mat[:, :3, 3]
        rot = mat[:, :3, :3]
        qwxyzw = np.zeros((len(mat), 7))
        for k in range(len(mat)):
            q = SciRot.from_matrix(rot[k]).as_quat()  # x y z w
            qwxyzw[k] = np.concatenate([xyz[k], [q[3], q[0], q[1], q[2]]])
        np.savetxt(path, qwxyzw, fmt="%.6f")
        print(f"[pose_graph] wrote {len(mat)} poses → {path}")
