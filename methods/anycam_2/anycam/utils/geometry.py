import torch

from minipytorch3d.rotation_conversions import (
    matrix_to_quaternion,
    quaternion_to_matrix,
    matrix_to_axis_angle,
    axis_angle_to_matrix,
)


def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    '''
    Spherical linear interpolation
    Args:
        t (float/np.ndarray): Float value between 0.0 and 1.0
        v0 (np.ndarray): Starting vector
        v1 (np.ndarray): Final vector
        DOT_THRESHOLD (float): Threshold for considering the two vectors as
                               colineal. Not recommended to alter this.
    Returns:
        v2 (np.ndarray): Interpolation vector between v0 and v1
    '''

    # Copy the vectors to reuse them later
    v0_copy = v0.clone()
    v1_copy = v1.clone()
    # Normalize the vectors to get the directions and angles
    v0 = v0 / v0.norm(dim=-1, keepdim=True)
    v1 = v1 / v1.norm(dim=-1, keepdim=True)

    # Dot product with the normalized vectors (can't use np.dot in W)
    dot = torch.sum(v0 * v1, dim=-1)

    # If absolute value of dot product is almost 1, vectors are ~colineal, so use lerp

    mask = torch.abs(dot) > DOT_THRESHOLD

    lerp_result = torch.lerp(v0_copy, v1_copy, t)

    # Calculate initial angle between v0 and v1
    theta_0 = torch.arccos(dot.unsqueeze(-1))
    sin_theta_0 = torch.sin(theta_0)
    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)
    # Finish the slerp algorithm
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    v2 = s0 * v0_copy + s1 * v1_copy

    res = v2

    res[mask] = lerp_result[mask]

    return res


def average_pose(poses, weight=0.5):
    rot_quat = matrix_to_quaternion(poses[..., :3, :3])

    rot_quat = slerp(weight, rot_quat[0], rot_quat[1])
    
    trans = poses[..., :3, 3]

    trans = (1 - weight) * trans[0] + weight * trans[1]

    rot_mat = quaternion_to_matrix(rot_quat)

    new_poses = torch.zeros_like(poses[0])

    new_poses[..., 3, 3] = 1
    new_poses[..., :3, :3] = rot_mat
    new_poses[..., :3, 3] = trans

    return new_poses


@torch.no_grad()
@torch.autocast(device_type="cuda", enabled=False)
def se3_ensure_numerical_accuracy(pose):
    """
    AnyCam internally uses float32 for pose representation. This can lead to numerical problems within 
    the evo package, which then detects the rotation part as not orthogonal. Therefore, we already convert 
    the pose to float64 and use SVD to ensure that the rotation part is orthogonal. This does not affect
    the final metrics, but ensures that evo does not throw an error. Check out this page for the 
    mathetmatical explanation: https://math.stackexchange.com/questions/2215359/showing-that-matrix-q-uvt-is-the-nearest-orthogonal-matrix-to-a
    :param pose: SE(3) pose
    :return: SE(3) pose with a valid rotation matrix
    """
    pose = pose.clone().to(torch.float64)
    rot = pose[..., :3, :3]
    U, S, Vh = torch.linalg.svd(rot)
    rot = U @ Vh
    pose[..., :3, :3] = rot
    return pose