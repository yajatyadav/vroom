from math import sin, cos

from einops import einsum
import torch
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


def transform_pts(pts: torch.Tensor, rel_pose: torch.Tensor) -> torch.Tensor:
    """Transform points by relative pose

    Args:
        pts (torch.Tensor): B, n_pts, 3
        rel_pose (torch.Tensor): B, 4, 4

    Returns:
        torch.Tensor: B, n_pts, 3
    """
    pts = torch.cat((pts, torch.ones_like(pts[..., :1])), dim=-1)
    return (pts @ rel_pose.transpose(-1, -2))[..., :3]


# TODO: unify
def distance_to_z(depths: torch.Tensor, projs: torch.Tensor):
    n, nv, h, w, _ = depths.shape
    device = depths.device

    inv_K = torch.inverse(projs)

    grid_x = (
        torch.linspace(-1, 1, w, device=device).view(1, 1, 1, -1).expand(-1, -1, h, -1)
    )
    grid_y = (
        torch.linspace(-1, 1, h, device=device).view(1, 1, -1, 1).expand(-1, -1, -1, w)
    )
    img_points = torch.stack((grid_x, grid_y, torch.ones_like(grid_x)), dim=2).expand(
        n, nv, -1, -1, -1
    )
    cam_points = (inv_K @ img_points.view(n, nv, 3, -1)).view(n, nv, 3, h, w)
    factors = cam_points[:, :, 2, :, :] / torch.norm(cam_points, dim=2)

    return depths * factors[..., None]


def z_to_distance(z: torch.Tensor, projs: torch.Tensor):
    n, nv, h, w = z.shape
    device = z.device

    inv_K = torch.inverse(projs)

    grid_x = (
        torch.linspace(-1, 1, w, device=device).view(1, 1, 1, -1).expand(-1, -1, h, -1)
    )
    grid_y = (
        torch.linspace(-1, 1, h, device=device).view(1, 1, -1, 1).expand(-1, -1, -1, w)
    )
    img_points = torch.stack((grid_x, grid_y, torch.ones_like(grid_x)), dim=2).expand(
        n, nv, -1, -1, -1
    )
    cam_points = (inv_K @ img_points.view(n, nv, 3, -1)).view(n, nv, 3, h, w)
    factors = cam_points[:, :, 2, :, :] / torch.norm(cam_points, dim=2)

    return z / factors


def azimuth_elevation_to_rotation(azimuth: float, elevation: float) -> torch.Tensor:
    rot_z = torch.tensor(
        [
            [cos(azimuth), -sin(azimuth), 0.0],
            [sin(azimuth), cos(azimuth), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rot_x = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos(azimuth), -sin(azimuth)],
            [0.0, sin(azimuth), cos(azimuth)],
        ]
    )
    return rot_x @ rot_z


def estimate_frustum_overlap(proj_source: torch.Tensor, pose_source: torch.Tensor, proj_target: torch.Tensor, pose_target: torch.Tensor, dist_lim=50):
    device = proj_source.device
    dtype = proj_source.dtype

    # Check which camera has higher z value in target coordinate system
    with autocast(enabled=False):
        src2tgt = torch.inverse(pose_target) @ pose_source

    for i in range(len(src2tgt)):
        if src2tgt[i, 2, 3] < 0:
            print("SWAP", i)
            proj_ = proj_target[i].clone()
            pose_ = pose_target[i].clone()
            proj_target[i] = proj_source[i]
            pose_target[i] = pose_source[i]
            proj_source[i] = proj_
            pose_source[i] = pose_

    points = torch.tensor([[
        [-1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, -1, 1, 1],
        [-1, -1, 1, 1],
    ]], device=device, dtype=dtype)

    with autocast(enabled=False):
        K_src_inv = torch.inverse(proj_source)
        K_tgt_inv = torch.inverse(proj_target)

    _ = K_src_inv.new_zeros(K_src_inv.shape[0], 4, 4)
    _[:, 3, 3] = 1
    _[:, :3, :3] = K_src_inv
    K_src_inv = _

    _ = K_tgt_inv.new_zeros(K_tgt_inv.shape[0], 4, 4)
    _[:, 3, 3] = 1
    _[:, :3, :3] = K_tgt_inv
    K_tgt_inv = _

    points_src = K_src_inv @ points.permute(0, 2, 1)
    points_tgt = K_tgt_inv @ points.permute(0, 2, 1)

    normals_tgt = torch.cross(points_tgt[..., :3, :], torch.roll(points_tgt[..., :3, :], shifts=-1, dims=-2), dim=-2)
    normals_tgt = normals_tgt / torch.norm(normals_tgt, dim=-2, keepdim=True)

    with autocast(enabled=False):
        src2tgt = torch.inverse(pose_target) @ pose_source

    base = src2tgt[:, :3, 3, None]
    points_src_tgt = src2tgt @ points_src

    dirs = points_src_tgt[..., :3, :] - base
    # dirs = dirs / torch.norm(dirs, dim=-2) #dirs should have z length 1

    dists = - (base[..., None] * normals_tgt[..., None, :]).sum(dim=-3) / (dirs[..., None] * normals_tgt[..., None, :]).sum(dim=-3).clamp_min(1e-4)

    # print(dists)

    # Ignore all non-positive
    mask = (dists <= 0) | (dists > dist_lim)
    dists[mask] = dist_lim

    # print(dists)

    dists = torch.min(dists, dim=-1)[0]

    mean_dist = dists.mean(dim=-1)

    # print(mean_dist, (torch.max(points_src[..., 0], dim=-1)[0] - torch.min(points_src[..., 0], dim=-1)[0]), (torch.max(points_src[..., 1], dim=-1)[0] - torch.min(points_src[..., 1], dim=-1)[0]))

    volume_estimate = \
        1/3 * \
        (torch.max(points_src[..., 0], dim=-1)[0] - torch.min(points_src[..., 0], dim=-1)[0]) * mean_dist * \
        (torch.max(points_src[..., 1], dim=-1)[0] - torch.min(points_src[..., 1], dim=-1)[0]) * mean_dist * \
        mean_dist

    return volume_estimate


def estimate_frustum_overlap_2(proj_source: torch.Tensor, pose_source: torch.Tensor, proj_target: torch.Tensor, pose_target: torch.Tensor, z_range=(3, 40), res=(8, 8, 16)):
    device = proj_source.device
    dtype = proj_source.dtype

    with autocast(enabled=False):
        K_src_inv = torch.inverse(proj_source)

    n = proj_source.shape[0]
    w, h, d = res

    pixel_width = 2 / w
    pixel_height = 2 / h

    x = torch.linspace(-1 + .5 * pixel_width, 1 - .5 * pixel_width, w, dtype=dtype, device=device).view(1, 1, 1, w).expand(n, d, h, w)
    y = torch.linspace(-1 + .5 * pixel_height, 1 - .5 * pixel_height, h, dtype=dtype, device=device).view(1, 1, h, 1).expand(n, d, h, w)
    z = torch.ones_like(x)

    xyz = torch.stack((x, y, z), dim=-1)
    xyz = K_src_inv @ xyz.reshape(n, -1, 3).permute(0, 2, 1)
    xyz = xyz.reshape(n, 3, d, h, w)

    # xyz = xyz * (1 / torch.linspace(1 / z_range[0], 1 / z_range[1], d, dtype=dtype, device=device).view(1, 1, d, 1, 1).expand(n, 1, d, h, w))
    xyz = xyz * torch.linspace(z_range[0], z_range[1], d, dtype=dtype, device=device).view(1, 1, d, 1, 1).expand(n, 1, d, h, w)

    xyz = torch.cat((xyz, torch.ones_like(xyz[:, :1])), dim=1)

    xyz = xyz.reshape(n, 4, -1)

    with autocast(enabled=False):
        src2tgt = torch.inverse(pose_target) @ pose_source

    xyz = src2tgt @ xyz

    # print(xyz)

    xyz = proj_target @ xyz[:, :3, :]

    xyz[:, :2] = xyz[:, :2] / xyz[:, 2:3, :]

    # print(xyz)

    valid = (xyz[:, 0].abs() < 1) & (xyz[:, 1].abs() < 1) & (xyz[:, 2].abs() > z_range[0])# & (xyz[:, 2].abs() < z_range[1])

    # print(valid)

    volume_estimate = valid.to(dtype).mean(-1)

    return volume_estimate


def compute_occlusions(flow0, flow1):
    n, _, h, w = flow0.shape
    device = flow0.device
    x = torch.linspace(-1, 1, w, device=device).view(1, 1, w).expand(1, h, w)
    y = torch.linspace(-1, 1, h, device=device).view(1, h, 1).expand(1, h, w)
    xy = torch.cat((x, y), dim=0).view(1, 2, h, w).expand(n, 2, h, w)
    flow0_r = torch.cat((flow0[:, 0:1, :, :] * 2 / w , flow0[:, 1:2, :, :] * 2 / h), dim=1)
    flow1_r = torch.cat((flow1[:, 0:1, :, :] * 2 / w , flow1[:, 1:2, :, :] * 2 / h), dim=1)

    xy_0 = xy + flow0_r
    xy_1 = xy + flow1_r

    xy_0 = xy_0.view(n, 2, -1)
    xy_1 = xy_1.view(n, 2, -1)

    ns = torch.arange(n, device=device, dtype=xy_0.dtype)
    nxy_0 = torch.cat((ns.view(n, 1, 1).expand(-1, 1, xy_0.shape[-1]), xy_0), dim=1)
    nxy_1 = torch.cat((ns.view(n, 1, 1).expand(-1, 1, xy_1.shape[-1]), xy_1), dim=1)

    mask0 = torch.zeros_like(flow0[:, :1, :, :])
    mask0[nxy_1[:, 0, :].long(), 0, ((nxy_1[:, 2, :] * .5 + .5) * h).round().long().clamp(0, h-1), ((nxy_1[:, 1, :] * .5 + .5) * w).round().long().clamp(0, w-1)] = 1

    mask1 = torch.zeros_like(flow1[:, :1, :, :])
    mask1[nxy_0[:, 0, :].long(), 0, ((nxy_0[:, 2, :] * .5 + .5) * h).round().long().clamp(0, h-1), ((nxy_0[:, 1, :] * .5 + .5) * w).round().long().clamp(0, w-1)] = 1

    return mask0, mask1


def align_rigid(
    p,
    q,
    weights,
):
    """Compute a rigid transformation that, when applied to p, minimizes the weighted
    squared distance between transformed points in p and points in q. See "Least-Squares
    Rigid Motion Using SVD" by Olga Sorkine-Hornung and Michael Rabinovich for more
    details (https://igl.ethz.ch/projects/ARAP/svd_rot.pdf).

    p: torch.Tensor (batch, n, 3)
        The source point set.
    q: torch.Tensor (batch, n, 3)
        The target point set.
    weights: torch.Tensor (batch, n)
        The weights to use when computing the transformation.
    """

    device = p.device
    dtype = p.dtype
    batch, _, _ = p.shape

    # 1. Compute the centroids of both point sets.
    weights_normalized = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
    p_centroid = (weights_normalized[..., None] * p).sum(dim=-2)
    q_centroid = (weights_normalized[..., None] * q).sum(dim=-2)

    # 2. Compute the centered vectors.
    p_centered = p - p_centroid[..., None, :]
    q_centered = q - q_centroid[..., None, :]

    # 3. Compute the 3x3 covariance matrix.
    covariance = (q_centered * weights[..., None]).transpose(-1, -2) @ p_centered

    # 4. Compute the singular value decomposition and then the rotation.

    with autocast(enabled=False):
        u, _, vt = torch.linalg.svd(covariance.to(torch.float32))
    
    s = torch.eye(3, dtype=dtype, device=device)
    s = s.expand((batch, 3, 3)).contiguous()
    s[..., 2, 2] = (u.det() * vt.det()).sign()
    rotation = u @ s @ vt

    # 5. Compute the optimal translation.
    translation = q_centroid - einsum(rotation, p_centroid, "... i j, ... j -> ... i")

    # Compose the results into a single transformation matrix.
    shape = (*rotation.shape[:-2], 4, 4)
    r = torch.eye(4, dtype=torch.float32, device=device).expand(shape).contiguous()
    r[..., :3, :3] = rotation
    t = torch.eye(4, dtype=torch.float32, device=device).expand(shape).contiguous()
    t[..., :3, 3] = translation

    return t @ r


def get_matches(p_depth, q_depth, q_flow, q_occ, proj):
    n, _, h, w = p_depth.shape
    device = p_depth.device

    x = torch.linspace(-1, 1, w, device=device).view(1, 1, 1, -1).expand(n, 1, h, w)
    y = torch.linspace(-1, 1, h, device=device).view(1, 1, -1, 1).expand(n, 1, h, w)
    xy = torch.cat((x, y), dim=1)

    xyz = torch.cat((xy, torch.ones_like(xy[:, :1])), dim=1)

    inv_proj = torch.inverse(proj)

    p_depths_unproj = (inv_proj @ xyz.view(n, 3, -1)).view(n, 3, h, w) * p_depth
    q_depths_unproj = (inv_proj @ xyz.view(n, 3, -1)).view(n, 3, h, w) * q_depth

    adj_xy = xy + q_flow

    valid = (adj_xy[:, :1, :, :] > -1) & (adj_xy[:, :1, :, :] < 1) & (adj_xy[:, 1:2, :, :] > -1) & (adj_xy[:, 1:2, :, :] < 1)
    valid = valid & (q_occ > .5)

    p_depths_unproj_resampled = F.grid_sample(p_depths_unproj, adj_xy.permute(0, 2, 3, 1), align_corners=False)

    p_pts = p_depths_unproj_resampled.view(n, 3, -1)
    q_pts = q_depths_unproj.view(n, 3, -1)
    valid = valid.view(n, -1)

    return p_pts, q_pts, valid, xyz


def get_valid_matches(flow, occ):
    device = flow.device
    n, _, h, w = flow.shape
    
    x = torch.linspace(-1, 1, w, device=device).view(1, 1, 1, -1).expand(n, 1, h, w)
    y = torch.linspace(-1, 1, h, device=device).view(1, 1, -1, 1).expand(n, 1, h, w)
    xy = torch.cat((x, y), dim=1)

    adj_xy = xy + flow

    valid = (adj_xy[:, :1, :, :] > -1) & (adj_xy[:, :1, :, :] < 1) & (adj_xy[:, 1:2, :, :] > -1) & (adj_xy[:, 1:2, :, :] < 1)
    valid = valid & (occ > .5)

    return valid


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


def resample(image, flow, subtrackt_pixel=True):
    n, c, h, w = image.shape
    xy = get_grid_xy(h, w, device=image.device)
    target = xy + flow
    target = target.permute(0, 2, 3, 1)
    
    image_resampled = F.grid_sample(image, target, mode="bilinear", align_corners=False, padding_mode="border")

    if subtrackt_pixel:
        p_w = .5 / w
        p_h = .5 / h
    else:
        p_w = 0
        p_h = 0

    mask = (target[..., 0] > -1 + p_w) & (target[..., 0] < 1 - p_w) & (target[..., 1] > -1 + p_h) & (target[..., 1] < 1 - p_h)

    return image_resampled, mask.view(n, 1, h, w)


def get_grid_xy(h, w, device="cpu", subtrackt_pixel=True, homogeneous=False):
    if subtrackt_pixel:
        p_w = .5 / w
        p_h = .5 / h
    else:
        p_w = 0
        p_h = 0

    x = torch.linspace(-1 + p_w, 1 - p_w, w, device=device).view(1, 1, 1, -1).expand(1, 1, h, w)
    y = torch.linspace(-1 + p_h, 1 - p_h, h, device=device).view(1, 1, -1, 1).expand(1, 1, h, w)

    xy = torch.cat((x, y), dim=1)

    if homogeneous:
        xy = torch.cat((xy, torch.ones_like(x)), dim=1)

    return xy


def induce_flow(depths, projs, poses):
    n, _, h, w = depths.shape
    device = depths.device

    xyz = get_grid_xy(h, w, device=device, homogeneous=True).expand(n, -1, -1, -1)
    xy = xyz[:, :2]

    inv_proj = torch.inverse(projs.reshape(-1, 3, 3))

    pts = (inv_proj @ xyz.view(n, 3, -1)) * depths.reshape(n, 1, -1)
    pts = torch.cat((pts, torch.ones_like(pts[:, :1])), dim=1)

    pts = poses @ pts

    pts = pts[:, :3] / pts[:, 2:3]

    pts = projs.reshape(-1, 3, 3) @ pts

    induced_flow = pts[:, :2].reshape(n, 2, h, w) - xy

    induced_flow = induced_flow.view(n, 2, h, w)

    return induced_flow


def compute_flow_errors(flow, induced_flow):
    error = (flow - induced_flow).abs().mean(dim=1, keepdim=True)
    return error


def pseudo_ransac_alignment(depths, flows, projs, poses, weights=None, num_parallel=10, num_points=64, inlier_threshold=0.01):
    n, _, h, w = depths.shape
    device = depths.device

    inv_projs = torch.inverse(projs)

    xyz = get_grid_xy(h, w, homogeneous=True, device=device).expand(n, -1, h, w).reshape(n, 3, -1)

    depths = depths.reshape(n, 1, -1)
    flows = flows.reshape(n, 2, -1)

    a = (projs @ poses[:, :3, :3] @ inv_projs) @ xyz
    b = projs @ poses[:, :3, 3:4]
    c = (xyz[:, :2] + flows)

    nom_x = (a[:, 0, :] - a[:, 2, :] * c[:, 0, :])
    nom_y = (a[:, 1, :] - a[:, 2, :] * c[:, 1, :])
    div_x = (b[:, 2, :] * c[:, 0, :] - b[:, 0, :])
    div_y = (b[:, 2, :] * c[:, 1, :] - b[:, 1, :])

    target_x = nom_x / div_x
    target_y = nom_y / div_y

    valid_x = (div_x.abs() > 5e-3) & (target_x > 0)
    valid_y = (div_y.abs() > 5e-3) & (target_y > 0)

    valid = torch.cat((valid_x, valid_y), dim=1)

    # If all pixels for an image are invalid, set all to invalid and hope that it defaults to no scale and shift
    full_invalid = (~valid).all(dim=-1)
    valid[full_invalid, :] = True

    target_x[full_invalid, :] = 0
    target_y[full_invalid, :] = 0

    weights = valid.float() * weights.reshape(n, 1, -1).expand(-1, 2, -1).reshape(n, -1)
    weights_norm = weights / weights.sum(dim=-1, keepdim=True)

    distribution = Categorical(probs=weights_norm)
    
    ids = distribution.sample((num_parallel, num_points))

    depth_1 = torch.cat((1 / depths, torch.ones_like(depths)), dim=1).view(n, 2, 1, -1).expand(-1, -1, 2, -1).reshape(n, 2, -1)

    A = depth_1.permute(0, 2, 1)
    B = torch.cat((target_x.reshape(n, -1), target_y.reshape(n, -1)), dim=1)[:, :, None]

    # (n, 2*h*w, 2) -> (n, num_parallel, num_points, 2)
    A_sampled = torch.gather(A, 1, ids[:, :, :, None].expand(-1, -1, -1, 2).reshape(n, num_parallel * num_points, 2))
    # (n, 2*h*w, 1) -> (n, num_parallel, num_points, 1)
    B_sampled = torch.gather(B, 1, ids[:, :, :, None].reshape(n, num_parallel * num_points, 1))

    # (n, num_parallel * num_points, 2) -> (n * num_parallel, num_points, 2)
    A_sampled = A_sampled.reshape(n * num_parallel, num_points, 2)
    # (n, num_parallel * num_points, 1) -> (n * num_parallel, num_points, 1)
    B_sampled = B_sampled.reshape(n * num_parallel, num_points, 1)

    lst_sq_out = torch.linalg.lstsq(A_sampled, B_sampled)
    X = lst_sq_out.solution

    scale_proposals = X[:, 0].reshape(n, num_parallel)
    shift_proposals = X[:, 1].reshape(n, num_parallel)
    
    scale_proposals = torch.cat((scale_proposals, torch.ones_like(scale_proposals[:, :1])), dim=-1)
    shift_proposals = torch.cat((shift_proposals, torch.zeros_like(shift_proposals[:, :1])), dim=-1)

    num_parallel = num_parallel + 1

    # (n, num_parallel, 1, h, w)
    depths = 1 / (
        1 / depths.reshape(n, 1, 1, h, w) * 
        scale_proposals.reshape(n, num_parallel, 1, 1, 1) + 
        shift_proposals.reshape(n, num_parallel, 1, 1, 1)
        ).clamp_min(1e-3)

    # (n, num_parallel, 1, h, w) -> (n * num_parallel, 1, h, w)
    induced_flows = induce_flow(
        depths.reshape(-1, 1, h, w),
        projs.reshape(n, 1, 3, 3).expand(-1, num_parallel, -1, -1).reshape(-1, 3, 3), 
        poses.reshape(n, 1, 4, 4).expand(-1, num_parallel, -1, -1).reshape(-1, 4, 4),
        ).reshape(n * num_parallel, 2, h, w)

    pixel_errors = compute_flow_errors(
        flows.reshape(n, 1, 2, h, w).expand(-1, num_parallel, -1, -1, -1).reshape(-1, 2, h, w),
        induced_flows,
        )
    
    inlier_ratios = (pixel_errors < inlier_threshold).float().mean(dim=(-3, -2, -1))
    inlier_ratios = inlier_ratios.reshape(n, num_parallel)

    best_id = inlier_ratios.argmax(dim=1)
    best_id[full_invalid] = num_parallel - 1
    
    accepted_depth = depths[torch.arange(n), best_id]
    accepted_induced_flows = induced_flows.reshape(n, num_parallel, 2, h, w)[torch.arange(n), best_id]

    return accepted_depth, accepted_induced_flows