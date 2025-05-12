from matplotlib import pyplot as plt
import numpy as np
import torch
import rerun as rr
import torch.nn.functional as F


from minipytorch3d.rotation_conversions import (
    matrix_to_quaternion,
    quaternion_to_matrix,
    matrix_to_axis_angle,
    axis_angle_to_matrix,
    standardize_quaternion,
)

from anycam.visualization.common import color_tensor


# This does not produce graph breaks when compiled with torch.compile()

def custom_matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = (
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    q_abs = torch.sqrt(q_abs.clamp_min(0.0))

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack(
                [q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1
            ),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack(
                [m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1
            ),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack(
                [m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1
            ),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack(
                [m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1
            ),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    # out = quat_candidates[
    #     F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    # ].reshape(batch_dim + (4,))
    out = torch.gather(quat_candidates, dim=-2, index=q_abs.argmax(dim=-1).view(*batch_dim, 1, 1).expand(*batch_dim, 1, 4)).squeeze(-2)

    return standardize_quaternion(out)


def compute_pixel_tracks(flow_occs, uncert, depth, grid_size = 8, track_len = 8, stride=4, is_backward=False, mask_depth=False, imgs=None, long_tracks=False):
    # flow_occs: (n, f, 3, h, w)
    # uncert: (n, f, 1, h, w)

    n = 1
    f, c, h, w = flow_occs.shape
    device = flow_occs.device

    flow_occs = flow_occs.view(n, f, c, h, w)
    uncert = uncert.view(n, f, 1, h, w)
    depth = depth.view(n, f, 1, h, w)

    if is_backward:
        flow_occs = flow_occs.flip(1)
        uncert = uncert.flip(1)
        depth = depth.flip(1)
    
    initial_depths = []
    pixel_tracks = []
    uncerts = []
    indices = []
    depths = []
    rgbs = []

    if long_tracks:
        track_until = f - 1
    else:
        track_until = f - track_len + 1

    for frame_idx in range(0, track_until, stride):
        x = torch.linspace(-1, 1, grid_size+2, device=device)[1:-1].view(1, -1).expand(grid_size, -1)
        y = torch.linspace(-1, 1, grid_size+2, device=device)[1:-1].view(-1, 1).expand(-1, grid_size)

        grid = torch.stack([x, y], dim=0).view(1, 2, grid_size, grid_size).expand(n, 2, grid_size, grid_size)
        grid = grid.permute(0, 2, 3, 1)

        grid_uncert = torch.zeros_like(grid[..., 0:1])

        curr_initial_depths = F.grid_sample(depth[:, frame_idx], grid, align_corners=False)
        curr_indices = [torch.zeros_like(grid[..., 0:1], dtype=torch.long) + frame_idx]
        curr_tracks = [grid]
        curr_uncerts = [grid_uncert]
        # curr_uncerts = [F.grid_sample(uncert[:, frame_idx], grid, align_corners=False).permute(0, 2, 3, 1)]
        curr_depths = [F.grid_sample(depth[:, frame_idx], grid, align_corners=False).permute(0, 2, 3, 1)]
        if imgs is not None:
            curr_rgbs = [F.grid_sample(imgs[frame_idx, None], grid, align_corners=False).permute(0, 2, 3, 1)]

        for i in range(track_len-1):
            if frame_idx + i < f - 1:
                uncert_ = uncert[:, frame_idx + i, :1]
                flow_ = flow_occs[:, frame_idx + i, :2]
                occ_ = flow_occs[:, frame_idx + i, 2:3]
                depth_ = depth[:, frame_idx + i]


                grid_flow = F.grid_sample(flow_, grid, align_corners=False).permute(0, 2, 3, 1)
                grid_occ = F.grid_sample(occ_, grid, align_corners=False, padding_mode="zeros").permute(0, 2, 3, 1)
                grid_uncert = F.grid_sample(uncert_, grid, align_corners=False).permute(0, 2, 3, 1)
                grid_depth = F.grid_sample(depth_, grid, align_corners=False).permute(0, 2, 3, 1)

                valid = torch.ones_like(grid[..., 0:1], dtype=bool)

                if mask_depth:
                    max_depth = 0.95 * torch.quantile(depth_.reshape(n, 1, -1), 0.95, dim=-1, keepdim=True).reshape(n, 1, 1, 1)
                    grid_valid_depth = grid_depth < max_depth
                    valid = valid & (grid_valid_depth > .5)

                valid = valid & (grid_occ > .5)

                grid = grid + grid_flow

                valid = valid & (grid[..., :1].abs() < .99) & (grid[..., 1:2].abs() < .99)

                grid_uncert = grid_uncert + curr_uncerts[-1]
                grid_uncert[~valid] = float("inf")

                curr_tracks.append(grid)
                curr_uncerts.append(grid_uncert)
                curr_indices.append(torch.zeros_like(grid[..., 0:1], dtype=torch.long) + frame_idx + i + 1)
                curr_depths.append(grid_depth)

                if imgs is not None:
                    curr_rgbs.append(F.grid_sample(imgs[frame_idx + i, None], grid, align_corners=False).permute(0, 2, 3, 1))

            else:
                invalid_uncerts = grid_uncert + float("inf")
                invalid_indices = torch.zeros_like(grid[..., 0:1], dtype=torch.long) + f - 1
                invalid_depths = torch.zeros_like(grid[..., 0:1])

                curr_tracks.append(grid.clone())
                curr_uncerts.append(invalid_uncerts)
                curr_indices.append(invalid_indices)
                curr_depths.append(invalid_depths)

                if imgs is not None:
                    curr_rgbs.append(curr_rgbs[-1].clone())

        curr_initial_depths = curr_initial_depths.reshape(n, grid_size ** 2)
        curr_tracks = torch.stack(curr_tracks, dim=1).reshape(n, track_len, grid_size ** 2, 2).permute(0, 2, 1, 3)
        curr_uncerts = torch.stack(curr_uncerts, dim=1).reshape(n, track_len, grid_size ** 2, 1).permute(0, 2, 1, 3)
        curr_indices = torch.stack(curr_indices, dim=1).reshape(n, track_len, grid_size ** 2, 1).permute(0, 2, 1, 3)
        curr_depths = torch.stack(curr_depths, dim=1).reshape(n, track_len, grid_size ** 2, 1).permute(0, 2, 1, 3)

        if imgs is not None:
            curr_rgbs = torch.stack(curr_rgbs, dim=1).reshape(n, track_len, grid_size ** 2, 3).permute(0, 2, 1, 3)

        initial_depths.append(curr_initial_depths)
        pixel_tracks.append(curr_tracks)
        uncerts.append(curr_uncerts)
        indices.append(curr_indices)
        depths.append(curr_depths)

        if imgs is not None:
            rgbs.append(curr_rgbs)

    initial_depths = torch.stack(initial_depths, dim=1)
    pixel_tracks = torch.stack(pixel_tracks, dim=1)
    uncerts = torch.stack(uncerts, dim=1)
    indices = torch.stack(indices, dim=1)
    depths = torch.stack(depths, dim=1)

    if imgs is not None:
        rgbs = torch.stack(rgbs, dim=1)

    if is_backward:
        indices = f - 1 - indices

    if imgs is not None:
        return initial_depths, pixel_tracks, uncerts, indices, depths, rgbs
    else:
        return initial_depths, pixel_tracks, uncerts, indices, depths


def get_corr_poses(indices, poses):
    n, wc, gs, tl, _ = indices.shape
    n, seq_len, _, _ = poses.shape

    indices = indices.reshape(n, -1, 1, 1).expand(-1, -1, 4, 4)

    corr_poses = torch.gather(poses, dim=1, index=indices)

    return corr_poses


def get_corr_scales_shifts(indices, scales, shifts):
    n, wc, gs, tl, _ = indices.shape
    seq_len, _ = scales.shape
    seq_len, _ = shifts.shape

    indices = indices.reshape(n, -1, 1)

    corr_scales = torch.gather(scales[None, ...], dim=1, index=indices)
    corr_shifts = torch.gather(shifts[None, ...], dim=1, index=indices)

    return corr_scales, corr_shifts


def param_to_pose(rot, t):
    n, seq_len, c = rot.shape

    if c == 3:
        rot_mat = axis_angle_to_matrix(rot)
    else:
        rot_mat = quaternion_to_matrix(rot)

    trans = t.view(n, seq_len, 3, 1)

    pose = torch.cat((rot_mat, trans), dim=-1)
    pose = torch.cat((pose, torch.tensor([[[0, 0, 0, 1]]], device=pose.device).expand(n, seq_len, -1, -1)), dim=-2)

    return pose


def pose_to_param(pose, representation="axis-angle"):
    n, seq_len, _, _ = pose.shape

    rot_mat = pose[:, :, :3, :3]
    t = pose[:, :, :3, 3]

    if representation == "axis-angle":
        rot = matrix_to_axis_angle(rot_mat)
    elif representation == "quaternion":
        rot = custom_matrix_to_quaternion(rot_mat)
    else:
        raise ValueError("Invalid representation")

    return rot, t


def make_normalized_proj(focal_length, aspect_ratio=1.0):
    proj = torch.eye(3, device=focal_length.device) * focal_length

    # proj[0, 0] /= aspect_ratio
    proj[1, 1] *= aspect_ratio

    proj[2, 2] = 1

    inv_proj = torch.eye(3, device=focal_length.device) / focal_length

    # inv_proj[0, 0] *= aspect_ratio
    inv_proj[1, 1] /= aspect_ratio

    inv_proj[2, 2] = 1

    return proj, inv_proj


def log_ba_state(ba_poses_c2w, points=None, imgs=None, timestep=0, point_colors=None, max_dist=-1):
    n, seq_len, _, _ = ba_poses_c2w.shape

    ba_poses_c2w = ba_poses_c2w.detach()

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    rr.set_time_sequence("timestep", timestep)

    
    for i in range(seq_len):
        rr.log(f"world/cam{i:04d}", 
               rr.Transform3D(
                   translation=ba_poses_c2w[0, i, :3, 3].cpu().numpy(), 
                   mat3x3=ba_poses_c2w[0, i, :3, :3].cpu().numpy(),
                   axis_length=0.01
                ),
                )

        if imgs is not None:
            h, w = imgs.shape[-2:]

            rr.log(f"world/cam{i:04d}/pinhole", rr.Pinhole(
                resolution=[w, h],
                focal_length=w,
                image_plane_distance=0.02,
            ), static=True)

            img = (imgs[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            rr.log(f"world/cam{i:04d}/pinhole/img", rr.Image(img).compress(jpeg_quality=95), static=True)
            
    if points is not None:
        points_ = points[0, :].T

        std, mean = torch.std_mean(points_, dim=0, keepdim=True)

        points_[:, 0].clamp_(mean[:, 0] - 10 * std[:, 0], mean[:, 0] + 10 * std[:, 0])
        points_[:, 1].clamp_(mean[:, 1] - 10 * std[:, 1], mean[:, 1] + 10 * std[:, 1])
        points_[:, 2].clamp_(mean[:, 2] - 10 * std[:, 2], mean[:, 2] + 10 * std[:, 2])

        if max_dist > 0:
            points_.clamp_(-max_dist, max_dist)

        points_ = points_.detach().cpu().numpy()

        if point_colors is not None:
            point_colors = (point_colors[0].detach().cpu().numpy() * 255).astype(np.uint8)
            rr.log("world/points", rr.Points3D(points_, colors=point_colors))
        else:
            rr.log("world/points", rr.Points3D(points_, colors=[[0, 255, 0]]))


def log_ba_imgs(imgs, uncertainties=None, tracks=None, timestep=0, frame_idx=0):
    seq_len, c, h, w = imgs.shape

    if tracks is not None:
        cmap = plt.get_cmap('hsv')

        indices, uncerts, pixel_tracks = tracks
        grid_total = indices.shape[2]
        grid_colors = np.array([cmap((i % grid_total) / grid_total) for i in range(grid_total)])[:, :3] * 255
        grid_colors = grid_colors.astype(np.uint8)

        grid_colors = torch.tensor(grid_colors).cuda()
        grid_colors = grid_colors.view(1, 1, grid_total, 1, 3).expand(*indices.shape[:2], -1, indices.shape[3], -1)

        grid_colors = color_tensor(-uncerts.clamp(0, .1), cmap="plasma", norm=True).view(*uncerts.shape[:-1], 3).cpu()
        indices = indices.cpu()

    if timestep == -1:
        for i in range(seq_len):
            img = (imgs[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            rr.set_time_sequence("timestep", i)
            rr.log(f"input/img", rr.Image(img).compress(jpeg_quality=95))

            if uncertainties is not None and i < len(uncertainties):
                uncert = uncertainties[i] 
                uncert = uncert / uncert.max()
                uncert = (uncert.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                rr.log(f"input/uncert", rr.Image(uncert).compress(jpeg_quality=95))

            if tracks is not None:

                mask = indices.reshape(-1) == i
                mask = mask & (uncerts.reshape(-1) < 1).cpu()

                tracks = pixel_tracks.reshape(-1, 2)[mask, :].cpu().numpy()
                colors = grid_colors.reshape(-1, 3)[mask, :].cpu().numpy()

                tracks[:, 0] = (tracks[:, 0] * .5 + .5) * w
                tracks[:, 1] = (tracks[:, 1] * .5 + .5) * h

                rr.log("input/tracks", rr.Points2D(tracks, colors=colors, radii=[1]))
    else:
        img = (imgs[frame_idx].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        rr.set_time_sequence("timestep", timestep)
        rr.log(f"input/img", rr.Image(img).compress(jpeg_quality=95))