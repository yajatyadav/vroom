from typing import overload
import numpy as np
import torch
from torchvision.transforms import functional as F

EPS = 1e-3


def normalize_calib(K: torch.Tensor, img_sizes: torch.Tensor) -> torch.Tensor:
    """Normalize the calibration matrices for fisheye cameras based on the image size

    Args:
        calib (torch.Tensor): B, n_views, 3, 3
        img_sizes (torch.Tensor): B, n_views, 2

    Returns:
        torch.Tensor: B, n_views 7
    """

    K[..., :2, :] = K[..., :2, :] / img_sizes.unsqueeze(-1) * 2.0
    K[..., :2, 2] = K[..., :2, 2] - 1.0

    return K


def unnormalize_calib(K: torch.Tensor, img_sizes: torch.Tensor) -> torch.Tensor:
    """Unnormalize the calibration matrices for fisheye cameras based on the image size

    Args:
        calib (torch.Tensor): B, n_views, 3, 3
        img_sizes (torch.Tensor): B, n_views, 2

    Returns:
        torch.Tensor: B, n_views 7
    """

    K[..., :2, 2] = K[..., :2, 2] + 1.0
    K[..., :2, :] = K[..., :2, :] * img_sizes.unsqueeze(-1) / 2.0

    return K


def pts_into_camera(pts: torch.Tensor, poses_w2c: torch.Tensor) -> torch.Tensor:
    """Project points from world coordinates into camera coordinate

    Args:
        pts (torch.Tensor): B, n_pts, 3
        poses_w2c (torch.Tensor): B, n_view, 4, 4

    Returns:
        torch.Tensor: B, n_views, n_pts, 3
    """

    # Add a singleton dimension to the input point cloud to match grid_f_poses_w2c shape
    pts = pts.unsqueeze(1)  # [B, 1, n_pts, 3]
    ones = torch.ones_like(
        pts[..., :1]
    )  ## Create a tensor of ones to add a fourth dimension to the point cloud for homogeneous coordinates
    pts = torch.cat(
        (pts, ones), dim=-1
    )  ## Concatenate the tensor of ones with the point cloud to create homogeneous coordinates
    return (poses_w2c[:, :, :3, :]) @ pts.permute(0, 1, 3, 2)


def project_to_image(
    pts: torch.Tensor, Ks: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project pts in camera coordinates into image coordinates.

    Args:
        pts (torch.Tensor): B, n_views, n_pts, 3
        Ks (torch.Tensor): B, n_views, 3, 3

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (B, n_views, n_pts, 2), (B, n_views, n_pts, 1)
    """
    pts = (Ks @ pts).permute(
        0, 1, 3, 2
    )  ## Apply the intrinsic camera parameters to the projected points to get pixel coordinates
    xy = pts[
        :, :, :, :2
    ]  ## Extract the x,y coordinates and depth value from the projected points
    z_ = pts[:, :, :, 2:3]

    xy = xy / z_.clamp_min(EPS)

    return xy, z_


def outside_frustum(
    xy: torch.Tensor,
    z: torch.Tensor,
    limits_x: tuple[float, float] | tuple[int, int] = (-1.0, 1.0),
    limits_y: tuple[float, float] | tuple[int, int] = (-1.0, 1.0),
    limit_z: float = EPS,
) -> torch.Tensor:
    """_summary_

    Args:
        xy (torch.Tensor): _description_
        z (torch.Tensor): _description_
        limits_x (tuple[float, float] | tuple[int, int], optional): _description_. Defaults to (-1.0, 1.0).
        limits_y (tuple[float, float] | tuple[int, int], optional): _description_. Defaults to (-1.0, 1.0).
        limit_z (float, optional): _description_. Defaults to EPS.

    Returns:
        torch.Tensor: _description_
    """
    return (
        (z <= limit_z)
        | (xy[..., :1] < limits_x[0])
        | (xy[..., :1] > limits_x[1])
        | (xy[..., 1:2] < limits_y[0])
        | (xy[..., 1:2] > limits_y[1])
    )


def normalize_camera_intrinsics(
    Ks: torch.Tensor | np.ndarray,
    img_sizes: torch.Tensor | list[int] | tuple[int, int],
) -> torch.Tensor | np.ndarray:
    """Normalize the camera intrinsics based on the image size

    Args:
        K (torch.Tensor): B, n_views, 3, 3
        img_sizes (torch.Tensor): B, n_views, 2

    Returns:
        torch.Tensor: B, n_views, 3, 3
    """
    if isinstance(Ks, np.ndarray):
        Ks_norm = Ks.copy()
    else:
        Ks_norm = Ks.clone()
    Ks_norm = torch.Tensor(Ks_norm)

    if isinstance(img_sizes, (list, tuple)):
        img_sizes = torch.tensor(img_sizes, device=Ks_norm.device)
        for _ in range(Ks_norm.ndim - 2):
            img_sizes = img_sizes.unsqueeze(0)

    Ks_norm[..., 0, :] = Ks_norm[..., 0, :] / img_sizes[..., 1:2] * 2.0
    Ks_norm[..., 1, :] = Ks_norm[..., 1, :] / img_sizes[..., 0:1] * 2.0
    Ks_norm[..., :2, 2] = Ks_norm[..., :2, 2] - 1.0

    if isinstance(Ks, np.ndarray):
        return Ks_norm.numpy()

    return Ks_norm


def unnormalize_camera_intrinsics(
    Ks_norm: torch.Tensor | np.ndarray,
    img_sizes: torch.Tensor | list[int] | tuple[int, int],
) -> torch.Tensor | np.ndarray:
    """Unnormalize the camera intrinsics based on the image size

    Args:
        K (torch.Tensor): [B, n_views, ]3, 3
        img_sizes (torch.Tensor): [B, n_views,] 2

    Returns:
        torch.Tensor: [B, n_views, ]3, 3
    """
    if isinstance(Ks_norm, np.ndarray):
        Ks = Ks_norm.copy()
    else:
        Ks = Ks_norm.clone()
    Ks = torch.Tensor(Ks)

    if isinstance(img_sizes, (list, tuple)):
        img_sizes = torch.tensor(img_sizes, device=Ks.device)
        for _ in range(Ks.ndim - 2):
            img_sizes = img_sizes.unsqueeze(0)

    Ks[..., :2, 2] = Ks[..., :2, 2] + 1.0
    Ks[..., 0, :] = Ks[..., 0, :] * img_sizes[..., 1:2] / 2.0
    Ks[..., 1, :] = Ks[..., 1, :] * img_sizes[..., 0:1] / 2.0
    # Ks[..., :2, :] = Ks[..., :2, :] * img_sizes.unsqueeze(-1) / 2.0
    if isinstance(Ks, np.ndarray):
        return Ks.numpy()

    return Ks


def resize_to_canonical_frame(
    img_data: torch.Tensor,
    K: torch.Tensor,
    canonical_K: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO: have different scale factors for x and y
    """Transform the input image into the canonical frame

    Args:
        img_data (torch.Tensor): B, C, H, W
        K (torch.Tensor): B, 3, 3
        canonical_K (torch.Tensor): 3, 3

    Returns:
        torch.Tensor, tuple[torch.Tensor, torch.Tensor]: B, C, H, W
    """

    # Compute the transformation matrix
    T = canonical_K.unsqueeze(0) @ torch.inverse(K)

    scale_factor = torch.sqrt(T[..., 0, 0] * T[..., 1, 1]).mean()  # 1

    K_new = K.clone()
    K_new[:, :2, :3] = K_new[:, :2, :3] * scale_factor.unsqueeze(-1).unsqueeze(-1)

    return (
        torch.nn.functional.interpolate(img_data, scale_factor=scale_factor.item()),
        K_new,
    )


def revert_to_original_frame(
    feature_maps: torch.Tensor, K: torch.Tensor, canonical_K: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform the input feature maps from the canonical frame to the original frame

    Args:
        feature_maps (torch.Tensor): B, C, H, W
        K (torch.Tensor): B, 3, 3
        canonical_K (torch.Tensor): B, 3, 3

    Returns:
        torch.Tensor: B, C, H, W
    """

    # Compute the transformation matrix
    T = K @ torch.inverse(canonical_K).unsqueeze(0)

    scale_factor = torch.sqrt(T[..., 0, 0] * T[..., 1, 1]).mean()  # 1

    K_new = K.clone()
    K_new[:, :2, :3] = K_new[:, :2, :3] * scale_factor.unsqueeze(-1).unsqueeze(-1)

    return torch.nn.functional.interpolate(feature_maps, scale_factor), K_new


def rescale_depth_map(
    depth_map: torch.Tensor, K: torch.Tensor, canonical_K: torch.Tensor
) -> torch.Tensor:
    """Rescale the depth map to the canonical frame

    Args:
        depth_map (torch.Tensor): B, n_views, H, W
        K (torch.Tensor): B, n_views, 3, 3
        canonical_K (torch.Tensor): B, n_views, 3, 3

    Returns:
        torch.Tensor: B, n_views, H, W
    """

    # Compute the transformation matrix
    T = canonical_K @ torch.inverse(K)

    scaling_factor = T[:, :, :2, :2].det().abs().sqrt()

    # Apply the transformation to the input depth map
    return depth_map * scaling_factor.unsqueeze(-1).unsqueeze(-1)


def _crop(
    img_data: torch.Tensor,
    K: torch.Tensor,
    xs: torch.Tensor,
    ys: torch.Tensor,
    crop_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    # Crop the input image
    img_data = torch.stack(
        [
            F.crop(image, int(y.item()), int(x.item()), crop_size[0], crop_size[1])
            for y, x, image in zip(ys, xs, img_data)
        ]
    )

    # Update the intrinsic camera parameters
    K[:, 0, 2] = K[:, 0, 2] - xs.to(K.device)
    K[:, 1, 2] = K[:, 1, 2] - ys.to(K.device)

    return img_data, K


def random_crop(
    img_data: torch.Tensor,
    K: torch.Tensor,
    crop_size: tuple[int, int],
    depth_maps: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly crop the input image and depth map

    Args:
        img_data (torch.Tensor): B, C, H, W
        K (torch.Tensor): B, 3, 3
        crop_size (tuple[int, int]): H, W

    Returns:
        torch.Tensor, tuple[torch.Tensor, torch.Tensor]: B, C, H, W
    """

    B, C, H, W = img_data.shape

    # Generate random crop coordinates
    x1 = torch.randint(0, W - crop_size[1] + 1, (B,))
    y1 = torch.randint(0, H - crop_size[0] + 1, (B,))
    x2 = x1 + crop_size[1]
    y2 = y1 + crop_size[0]

    return _crop(img_data, K, x1, y1, crop_size)


def center_crop(
    img_data: torch.Tensor,
    K: torch.Tensor,
    crop_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Central crop the input image and depth map

    Args:
        image (torch.Tensor): B, C, H, W
        K (torch.Tensor): B, 3, 3
        crop_size (tuple[int, int]): H, W

    Returns:
        torch.Tensor, tuple[torch.Tensor, torch.Tensor]: B, C, H, W
    """

    B, C, H, W = img_data.shape
    center_points = (torch.tensor([H, W]) / 2)[None].expand(B, 2).long()
    x1 = (center_points[:, 1] - crop_size[1] / 2).long()
    y1 = (center_points[:, 0] - crop_size[0] / 2).long()
    x2 = x1 + crop_size[1]
    y2 = y1 + crop_size[0]

    return _crop(img_data, K, x1, y1, crop_size)


def no_crop(
    img_data: torch.Tensor, K: torch.Tensor, crop_size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """No crop the input image and depth map

    Args:
        img_data (torch.Tensor): B, C, H, W
        K (torch.Tensor): B, 3, 3

    Returns:
        torch.Tensor, tuple[torch.Tensor, torch.Tensor]: B, C, H, W
    """
    return img_data, K
