from subprocess import check_output
import uuid
import numpy as np
import pycolmap
from pathlib import Path
import cv2

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


def calib_to_mat(params):
    proj = np.eye(3)
    proj[0, 0] = params[0]
    proj[1, 1] = params[1]
    proj[0, 2] = params[2]
    proj[1, 2] = params[3]
    return proj


def get_poses_from_droidslam(
    imgs,
    proj,
    droidslam_command_template,
    scene_name,
    out_dir=None,
    tmp_dir="/tmp/droidslam-io"
):
    tmp_dir = tmp_dir + str(uuid.uuid4())

    # Create a temporary directory
    tmp_dir = Path(tmp_dir) / scene_name
    tmp_dir.mkdir(exist_ok=True, parents=True)

    out_dir = Path(out_dir) / scene_name
    out_dir.mkdir(exist_ok=True, parents=True)

    img_dir = tmp_dir / "images"
    img_dir.mkdir(exist_ok=True)

    print("Saving images to tmp dir")

    for i, img in enumerate(imgs):
        cv2.imwrite(str(img_dir / f"{i:04d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    calib_params = np.array([proj[0, 0], proj[1, 1], proj[0, 2], proj[1, 2]])

    np.savetxt(tmp_dir / "calib.txt", calib_params)

    droidslam_output = Path(tmp_dir) / "out"

    # Run DROID-SLAM
    print("Running DROID-SLAM command")

    droidslam_command = droidslam_command_template.format(imagedir=str(img_dir), calib=str(tmp_dir / "calib.txt"), reconstruction_path=str(droidslam_output))

    print(droidslam_command)

    try:
        droidslam_cli_output = check_output(droidslam_command, shell=True)

        if out_dir is not None:
            # Copy results to out dir
            copy_command = f"cp -r {droidslam_output}/* {out_dir}"
            print(copy_command)
            check_output(copy_command, shell=True)
        else:
            out_dir = tmp_dir

        # Load poses
        print("Loading poses")

        poses = np.load(out_dir / "poses_full.npy")
        projs = np.load(out_dir / "intrinsics.npy")

        poses = np.stack([transform44(pose) for pose in poses])
        projs = np.stack([calib_to_mat(proj) for proj in projs])

        print(poses.shape)
        print(projs.shape)

    except:
        poses, projs = None, None
        droidslam_cli_output = None

    # Clear tmp dir
    rm_command = f"rm -r {tmp_dir}"

    print(rm_command)
    check_output(rm_command, shell=True)

    return poses, projs, droidslam_cli_output
