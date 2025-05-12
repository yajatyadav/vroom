# --------------------------------------------------------
# Helper for automatic loop-closure discovery
# --------------------------------------------------------
#
# Usage:
#     from dust3r.cloud_opt.loop_search import build_loop_edges
#     loop_edges = build_loop_edges(rgb_frames, poses, fps=30)
#     # loop_edges : list[(i, j, sim3_(4,4) torch, weight float)]
# --------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
from torchvision import transforms as T
from scipy.spatial.transform import Rotation

# ------------------------------------------------------------------ #
# 1. GeM descriptor model (128-D)
# ------------------------------------------------------------------ #
class GeM(nn.Module):
    """GeM pooling as in CVPR'18."""
    def __init__(self, p: float = 3.0, dim: int = 128):
        super().__init__()
        self.backbone = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # ⇢ (B,2048,H/32,W/32)
        self.p = nn.Parameter(torch.tensor(p))
        self.fc = nn.Linear(2048, dim, bias=False)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)                               # (B,2048,h,w)
        x = torch.clamp(x, min=1e-6).pow(self.p).mean([2,3]).pow(1/self.p)
        x = nn.functional.normalize(self.fc(x))
        return x                                           # (B,dim)


_gem_trans = T.Compose([
    T.ToTensor(),
    T.Resize(256, antialias=True),
    T.CenterCrop(224),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])


@torch.inference_mode()
def _compute_descriptors(frames: List[np.ndarray],
                         device: str = 'cuda') -> torch.Tensor:
    """frames: list of uint8 H×W×3 images → (N,128) float32 descriptors"""
    model = GeM().to(device).eval()
    descs = []
    bs = 32
    for i in range(0, len(frames), bs):
        batch = torch.stack([_gem_trans(im) for im in frames[i:i+bs]]).to(device)
        descs.append(model(batch).cpu())
    return torch.cat(descs, 0)            # (N,128)


# ------------------------------------------------------------------ #
# 2. k-NN proposal with FAISS (falls back to empty list if missing)
# ------------------------------------------------------------------ #
def _faiss_knn(desc: np.ndarray, k: int = 20) -> List[List[int]]:
    try:
        import faiss  # type: ignore
    except ImportError:
        print('[loop_search] FAISS not available – skipping loop search')
        return [[] for _ in range(len(desc))]
    index = faiss.IndexFlatL2(desc.shape[1])
    index.add(desc.astype('float32'))
    _, I = index.search(desc.astype('float32'), k+1)  # +1 → self hit
    return [row[1:].tolist() for row in I]            # drop self


# ------------------------------------------------------------------ #
# 3. SuperGlue verification  + Umeyama-RANSAC Sim(3) estimation
# ------------------------------------------------------------------ #
def _load_superglue(device: str = 'cuda'):
    try:
        from third_party.superglue.model import SuperGlue, cfg as SGcfg  # type: ignore
    except Exception:
        return None
    cfg = SGcfg(default={'weights': 'indoor'})
    return SuperGlue(cfg).to(device).eval()


@torch.inference_mode()
def _superglue_verify(imA: np.ndarray, imB: np.ndarray,
                      sg, device='cuda',
                      min_matches: int = 30) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return two matched 2-D keypoint arrays if verified, else None."""
    # lazy import of SuperPoint
    from third_party.superglue.utils import extract_spp_kpdesc  # type: ignore
    kpa, desc_a = extract_spp_kpdesc(imA, device)
    kpb, desc_b = extract_spp_kpdesc(imB, device)
    if kpa.shape[0] < 50 or kpb.shape[0] < 50:
        return None
    data = {
        'keypoints0': torch.from_numpy(kpa)[None].to(device),
        'keypoints1': torch.from_numpy(kpb)[None].to(device),
        'descriptors0': torch.from_numpy(desc_a)[None].to(device),
        'descriptors1': torch.from_numpy(desc_b)[None].to(device),
        'scores0': torch.ones(kpa.shape[0], device=device)[None],
        'scores1': torch.ones(kpb.shape[0], device=device)[None],
    }
    out = sg(data)
    msk = out['matches0'][0] > -1
    mkA = kpa[msk.cpu().numpy()]
    mkB = kpb[out['matches0'][0][msk].cpu().numpy()]
    if mkA.shape[0] < min_matches:
        return None
    return mkA, mkB


def _umeyama_sim3(src: np.ndarray, dst: np.ndarray, scale: bool = True):
    assert src.shape == dst.shape
    mu_a = src.mean(0)
    mu_b = dst.mean(0)
    src_c = src - mu_a
    dst_c = dst - mu_b
    H = src_c.T @ dst_c / src.shape[0]
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    if scale:
        s = S.sum() / (src_c ** 2).sum()
    else:
        s = 1.0
    t = mu_b - s * R @ mu_a
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T, s


def build_loop_edges(frames: List[np.ndarray],
                     poses_w2c: np.ndarray,
                     fps: float = 30.0,
                     k: int = 20,
                     min_dt: float = 15.0,
                     device: str = 'cuda') \
                     -> List[Tuple[int, int, torch.Tensor, float]]:
    """
    Returns a list of robust Sim(3) loop edges:
        (idx_a, idx_b, sim3_(4,4) torch, weight)
    """
    if len(frames) < 4:
        return []

    # 1) global descriptors ------------------------------------------------
    desc = _compute_descriptors(frames, device=device).numpy()  # (N,128)
    knn = _faiss_knn(desc, k=k)

    # 2) SuperGlue verification -------------------------------------------
    sg = _load_superglue(device)
    edges = []
    min_frames_apart = int(min_dt * fps)

    for i, nbrs in enumerate(knn):
        for j in nbrs:
            if abs(i - j) <= min_frames_apart:
                continue
            if sg is None:
                continue  # cannot verify
            match = _superglue_verify(frames[i], frames[j], sg, device=device)
            if match is None:
                continue
            kpA, kpB = match
            # lift to 3-D with depth = 1 (homog rays) --------------
            K = np.eye(3)
            raysA = np.concatenate([kpA, np.ones((kpA.shape[0], 1))], 1)
            raysB = np.concatenate([kpB, np.ones((kpB.shape[0], 1))], 1)
            # 3) Umeyama-RANSAC Sim(3) -----------------------------
            # quick 4-pt RANSAC
            best_inl, best_T = 0, None
            for _ in range(256):
                subset = np.random.choice(len(raysA), 4, replace=False)
                T_est, _ = _umeyama_sim3(raysA[subset], raysB[subset])
                proj = (T_est[:3, :3] @ raysA.T + T_est[:3, 3:4]).T
                err = np.linalg.norm(proj / proj[:, 2:3] - raysB, axis=1)
                inl = (err < 2).sum()
                if inl > best_inl:
                    best_inl, best_T = inl, T_est
            if best_inl < 40:
                continue
            # weight = inlier ratio
            weight = best_inl / len(raysA)
            edges.append((i, j,
                          torch.tensor(best_T, dtype=torch.float32),
                          float(weight)))
    print(f'[loop_search] found {len(edges)} loop edges')
    return edges
