from subprocess import check_output
import uuid
import numpy as np
import pycolmap
from pathlib import Path
import cv2
import os
import shutil


def read_sparse_reconstruction(rec_path: Path) -> None:
    print("Reading sparse COLMAP reconstruction")
    reconstruction = pycolmap.Reconstruction(rec_path)
    cameras = reconstruction.cameras
    images = reconstruction.images
    points3D = reconstruction.points3D

    poses = []
    projs = []

    for image in sorted(images.values(), key=lambda im: im.name):
        camera = cameras[image.camera_id]

        pose = np.eye(4)
        pose[:3, 3] = image.cam_from_world.translation
        pose[:3, :3] = image.cam_from_world.rotation.matrix()
        pose = np.linalg.inv(pose)

        cam_params = camera.params

        proj = np.eye(3)

        if len(cam_params) == 4:
            proj[0, 0] = cam_params[0]
            proj[1, 1] = cam_params[1]
            proj[0, 2] = cam_params[2]
            proj[1, 2] = cam_params[3]
        elif len(cam_params) == 3:
            proj[0, 0] = cam_params[0]
            proj[1, 1] = cam_params[0]
            proj[0, 2] = cam_params[1]
            proj[1, 2] = cam_params[2]

        poses.append(pose)
        projs.append(proj)

    return poses, projs


def export_to_colmap(trajectory, proj, imgs=None, out_dir=None):
    """
    Export camera trajectory in COLMAP format.
    
    Args:
        trajectory: List of 4x4 camera poses (world-to-camera)
        proj: 3x3 camera projection matrix
        imgs: Optional list of images corresponding to poses
        out_dir: Output directory to save COLMAP reconstruction
        
    Returns:
        Path to the COLMAP reconstruction
    """
    if out_dir is None:
        out_dir = Path("/tmp") / f"colmap-export-{uuid.uuid4()}"
    else:
        out_dir = Path(out_dir)
    
    out_dir.mkdir(exist_ok=True, parents=True)
    sparse_dir = out_dir / "sparse" / "0"
    sparse_dir.mkdir(exist_ok=True, parents=True)
    
    # Create a new COLMAP reconstruction
    reconstruction = pycolmap.Reconstruction()
    
    # Add camera
    height, width = 0, 0
    if imgs is not None and len(imgs) > 0:
        height, width = imgs[0].shape[:2]
    else:
        # Default size if images not provided
        height, width = 480, 640
    
    # Use SIMPLE_PINHOLE camera model
    camera_id = 0
    camera = pycolmap.Camera(
        camera_id=camera_id,
        model="SIMPLE_PINHOLE",
        width=width,
        height=height,
        params=[proj[0, 0], proj[0, 2], proj[1, 2]]
    )
    reconstruction.add_camera(camera)
    
    # Add images and poses
    for i, pose in enumerate(trajectory):
        # Convert from world-to-camera to camera-to-world
        camera_to_world = np.linalg.inv(pose)
        rotation = pycolmap.Rotation3d(camera_to_world[:3, :3])
        translation = camera_to_world[:3, 3]
               
        image = pycolmap.Image(
                name=f"{i:06d}.png",
                image_id=i,
                camera_id=camera_id,
                cam_from_world=pycolmap.Rigid3d(rotation, translation)
            )

        # Add image to reconstruction
        reconstruction.add_image(image)
        
        # Set image as registered
        reconstruction.register_image(i)
    
    # Save images if provided
    if imgs is not None:
        images_dir = out_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        for i, img in enumerate(imgs):
            img = (img * 255).astype(np.uint8)
            cv2.imwrite(str(images_dir / f"{i:06d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    # Write reconstruction to disk
    reconstruction.write(sparse_dir)
    
    return sparse_dir


def get_poses_from_colmap(
    imgs,
    colmap_command_template,
    scene_name,
    out_dir=None,
    tmp_dir="/tmp/colmap-io"
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

    # Run COLMAP
    print("Running COLMAP command")

    colmap_command = colmap_command_template.format(str(tmp_dir))

    print(colmap_command)

    try:
        colmap_cli_output = check_output(colmap_command, shell=True)

        if out_dir is not None:
            # Copy results to out dir
            colmap_output = Path(tmp_dir) / "sparse"

            copy_command = f"cp -r {colmap_output} {out_dir}"

            print("Copying results to out dir")
            print(copy_command)

            check_output(copy_command, shell=True)
        else:
            out_dir = tmp_dir

        if (out_dir / "sparse" / "0").exists():
            out_dir = out_dir / "sparse" / "0"
        else:
            out_dir = out_dir / "sparse"

        # Load poses
        poses, projs = read_sparse_reconstruction(out_dir)

    except:
        poses, projs = None, None
        colmap_cli_output = None

    print("Clearing directory")

    # Clear tmp dir
    check_output(f"rm -r {tmp_dir}", shell=True)

    return poses, projs, colmap_cli_output
