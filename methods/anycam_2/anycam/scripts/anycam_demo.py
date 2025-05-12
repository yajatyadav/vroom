import sys
import os
import uuid

import cv2

sys.path.append(".")
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from moviepy import VideoFileClip
import rerun as rr


from anycam.loss import make_loss
from anycam.trainer import AnyCamWrapper
from anycam.common.geometry import get_grid_xy
from anycam.utils.geometry import se3_ensure_numerical_accuracy
from anycam.visualization.common import color_tensor




def load_video(video_path):
    video = VideoFileClip(video_path)
    frames = [frame for frame in video.iter_frames()]
    frames = [frame.astype(np.float32) / 255.0 for frame in frames]
    fps = video.fps
    return frames, fps


def subsample_frames(frames, original_fps=None, target_fps=0):
    """
    Subsample frames to achieve target framerate
    
    Args:
        frames: List of frames
        original_fps: Original framerate of the video (if known)
        target_fps: Target framerate (0 or None means use all frames)
        
    Returns:
        List of subsampled frames
    """
    if not frames or target_fps <= 0 or not original_fps:
        return frames
        
    # Calculate the stride to achieve target fps
    stride = max(1, round(original_fps / target_fps))
    
    return frames[::stride]


def load_frames(image_path):
    frames = []

    for filename in tqdm(list(sorted(os.listdir(image_path)))):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(image_path, filename)
            frame = cv2.imread(file_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            frame = frame.astype(np.float32) / 255.0

            frames.append(frame)

    return frames, None


def format_frames(frames, target_size=336):
    height, width = frames[0].shape[:2]

    if height < width:
        new_height = target_size
        new_width = int((target_size / height) * width)
    else:
        new_width = target_size
        new_height = int((target_size / width) * height)
    
    frames = [cv2.resize(frame, (new_width, new_height)) for frame in frames]

    return frames


def load_anycam(model_path, checkpoint=None):
    config = OmegaConf.load(model_path / "training_config.yaml")

    prefix = "training_checkpoint_"
    ckpts = Path(model_path).glob(f"{prefix}*.pt")

    model_conf = config["model"]
    model_conf["use_provided_flow"] = False
    model_conf["train_directions"] = "forward"

    model = AnyCamWrapper(model_conf)

    criterion = [make_loss(cfg) for cfg in config.get("loss", [])][0]

    training_steps = [int(ckpt.stem.split(prefix)[1]) for ckpt in ckpts]

    if training_steps:
        if checkpoint is None:
            ckpt_path = f"{prefix}{max(training_steps)}.pt"
        else:
            ckpt_path = checkpoint

        ckpt_path = Path(model_path) / ckpt_path

        print(ckpt_path)

        cp = torch.load(ckpt_path, map_location="cpu")

        model.load_state_dict(cp["model"], strict=False)

    return model, criterion


def process_video(model, criterion, frames, config=None, ba_refinement=True):
    """
    Process a video by fitting the AnyCam model to the provided frames.
    
    Args:
        model: The AnyCam model
        criterion: The loss criterion
        frames: List of frames as numpy arrays with shape (H,W,3) and values in [0,1]
        config: Optional configuration dictionary for the fit_video function
               If None, default configuration will be used
        ba_refinement: Whether to perform bundle adjustment refinement (default: True)
    
    Returns:
        trajectory: The estimated camera trajectory
        proj: The camera projection matrix
        extras_dict: Additional information from the fitting process
        ba_extras: Bundle adjustment extra information
    """
    from dotdict import dotdict
    from anycam.scripts.fit_video import fit_video
    
    # Default configuration if none provided
    if config is None:
        default_config = {
            "with_rerun": False,
            "do_ba_refinement": ba_refinement,
            "prediction": {
                "model_seq_len": 100,
                "shift": 599,
                "square_crop": True,
                "return_all_uncerts": False,
            },
            "ba_refinement": {
                "with_rerun": False,
                "max_uncert": 0.05,
                "lambda_smoothness": 0.1,
                "long_tracks": True,
                "n_steps_last_global": 15_000,
            },
            "ba_refinement_level": 2,
            "dataset": {
                "image_size": [336, None]
            }
        }
        config = dotdict(default_config)
    elif not isinstance(config, dotdict):
        config = dotdict(config)
    
    # Ensure the BA refinement setting is applied to the config
    config.do_ba_refinement = ba_refinement

    print(f"Processing {len(frames)} frames...")
    print(f"Bundle adjustment refinement: {'Enabled' if ba_refinement else 'Disabled'}")
    
    # Run fit_video function
    trajectory, proj, extras_dict, ba_extras = fit_video(
        config,
        model,
        criterion,
        frames,
        return_extras=True,
    )
    
    print("Finished processing video")
    return trajectory, proj, extras_dict, ba_extras


def plot_to_rerun(
        trajectory, 
        depths, 
        imgs, 
        proj,
        uncertainties=None, 
        subsample_pts=1, 
        radii=1.5, 
        uncertainty_thresh=-1, 
        max_depth=-1, 
        filter_depth_threshold=0.1,
        image_plane_distance=0.05,
        keyframes=None,
        rerun_mode="spawn",
        rerun_address="localhost:8787",
        ):
    
    h, w = imgs[0].shape[:2]

    def filter_depth(depth, threshold=0.1):
        _, h, w = depth.shape

        depth = depth.clone()[None, ...]
        median = torch.median(depth)
        
        depth_grad = torch.stack(torch.gradient(depth, dim=(-2, -1))).norm(dim=0)

        mask = depth_grad < median * threshold

        return mask
    
    def lift_image(img, depth, pose, proj):
        h, w = img.shape[:2]
        device = depth.device

        proj = torch.tensor(proj, device=device).float()

        proj[0, 0] = proj[0, 0] / w * 2
        proj[1, 1] = proj[1, 1] / h * 2
        proj[0, 2] = proj[0, 2] / w * 2 - 1
        proj[1, 2] = proj[1, 2] / h * 2 - 1

        inv_proj = torch.inverse(proj)

        pts = get_grid_xy(h, w, homogeneous=True).reshape(3, h*w).to(device)
        pts = inv_proj @ pts
        pts = pts * depth.view(1, -1).to(device)
        pts = torch.cat((pts, torch.ones(1, h*w, device=device)), dim=0)
        pts = pose.to(pts.dtype) @ pts
        pts = pts[:3, :].T

        colors = torch.tensor(img.reshape(-1, 3)).to(device)

        return pts, colors
    
    imgs = np.array(imgs)

    # Initialize rerun with appropriate mode
    if rerun_mode == "spawn":
        rr.init("AnyCam Demo", recording_id=uuid.uuid4(), spawn=True)
    elif rerun_mode == "connect":
        rr.init("AnyCam Demo", recording_id=uuid.uuid4(), spawn=False)
        print(f"Connecting to existing Rerun server.")
        rr.connect(rerun_address)
        # rr.connect_tcp(addr="localhost:9090")
    else:
        raise ValueError(f"Unsupported rerun mode: {rerun_mode}. Use 'spawn' or 'connect'.")
    
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    rr.log("world/scene", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    blueprint = rr.blueprint.Blueprint(
        rr.blueprint.Horizontal(
            rr.blueprint.Spatial3DView(origin="/world/scene"),
            rr.blueprint.Vertical(
                rr.blueprint.Spatial2DView(origin="/world/scene/active_cam/input"),
                rr.blueprint.Spatial2DView(origin="/world/scene/active_cam/uncertainty"),
            ),
        ),
    )
    rr.send_blueprint(blueprint, make_active=True)

    for id in range(len(trajectory)):
        rr.set_time_sequence("step", id)


        pose = trajectory[id]
        rot = pose[:3, :3].cpu().numpy()


        rr.log(f"world/scene/active_cam", rr.Pinhole(
            resolution=[w, h],
            focal_length=float(proj[0, 0]),
            image_plane_distance=image_plane_distance, 
        ), static=True)
        rr.log(f"world/scene/active_cam", rr.Transform3D(translation=pose[:3, 3].cpu(), mat3x3=rot, axis_length=0.01))

        rr.log("world/scene/cam_traj", rr.LineStrips3D([pose[:3, 3].cpu().numpy().tolist() for pose in trajectory[:id+1]], colors=[(0, 255, 0)]), static=False)

        rr.log("world/scene/active_cam/input", rr.Image((imgs[id] * 255).astype(np.uint8)).compress(jpeg_quality=95))
        
        rr.log("world/scene", rr.Transform3D(translation=pose[:3, 3].cpu(), mat3x3=rot, axis_length=0, from_parent=True))

        if keyframes is None or id in keyframes:

            kid = keyframes.index(id)

            pts, colors = lift_image(torch.tensor(imgs[id]).cuda(), depths[kid].cuda(), trajectory[id].cuda(), proj)
            mask = filter_depth(depths[kid].cuda(), threshold=filter_depth_threshold)

            mask = mask.view(-1)

            if max_depth > 0:
                mask = mask & depths[kid].view(-1)

            pts = pts[mask, :]
            colors = colors[mask, :]

            pts = pts[subsample_pts//2::subsample_pts]
            colors = colors[subsample_pts//2::subsample_pts]
            colors = (colors * 255).clamp(0, 255).to(torch.uint8)

            rr.log(f"world/scene/active_points", rr.Points3D(pts[:, :3].cpu().numpy(), colors=colors[:, :3].cpu().numpy(), radii=rr.Radius.ui_points([radii]),))

            if uncertainties is not None:
                uncertainty_img = color_tensor((uncertainties[kid] / uncertainty_thresh).clamp(0, 1), cmap="plasma", norm=False)[0]
                uncertainty_img = uncertainty_img.cpu().numpy()
                uncertainty_img = (uncertainty_img * 255).astype(np.uint8)

                rr.log(f"world/scene/active_cam/uncertainty", rr.Image(uncertainty_img).compress(jpeg_quality=95))



@hydra.main(version_base=None, config_name=None)
def main(cfg: DictConfig):
    """
    AnyCam demo script for processing videos and extracting 3D information.
    
    Example usage:
    - Process video: python anycam_demo.py input_path=/path/to/video output_path=/path/to/output
    - Process images: python anycam_demo.py input_path=/path/to/images_folder output_path=/path/to/output
    - Visualize with rerun: python anycam_demo.py input_path=/path/to/video visualize=true
    - Connect to existing rerun server: python anycam_demo.py input_path=/path/to/video visualize=true rerun_mode=connect rerun_address=localhost:8787
    - Export to COLMAP: python anycam_demo.py input_path=/path/to/video export_colmap=true output_path=/path/to/colmap_output
    - Disable BA refinement: python anycam_demo.py input_path=/path/to/video ba_refinement=false
    - Subsample frames: python anycam_demo.py input_path=/path/to/video fps=10
    
    Config parameters:
    - input_path: Path to video or directory of images
    - output_path: Path to save outputs
    - model_path: Path to model (optional)
    - checkpoint: Specific checkpoint to use (optional)
    - visualize: Whether to visualize results with rerun (boolean)
    - rerun_mode: Mode to use for rerun visualization ('spawn' or 'connect', default: 'spawn')
    - rerun_address: Address to connect to when using rerun_mode=connect (default: localhost:8787)
    - export_colmap: Whether to export to COLMAP format (boolean)
    - image_size: Target image size for processing (default: 336)
    - ba_refinement: Whether to perform bundle adjustment refinement (default: True)
    - fps: Target frames per second (default: 0, use all frames)
    - vis: Visualization parameters subconfig with the following options:
        - subsample_pts: Point sampling rate (default: 1)
        - radii: Point radius for visualization (default: 1.5)
        - uncertainty_thresh: Threshold for uncertainty visualization (default: 0.05)
        - max_depth: Maximum depth value to consider (default: -1, no limit)
        - filter_depth_threshold: Threshold for depth filtering (default: 0.1)
        - image_plane_distance: Distance of image plane in visualization (default: 0.05)
    """
    input_path = cfg.get("input_path", None)
    output_path = cfg.get("output_path", None)
    model_path = cfg.get("model_path", None)
    checkpoint = cfg.get("checkpoint", None)
    visualize = cfg.get("visualize", False)
    rerun_mode = cfg.get("rerun_mode", "spawn")
    rerun_address = cfg.get("rerun_address", "localhost:8787")
    export_colmap = cfg.get("export_colmap", False)
    image_size = cfg.get("image_size", 336)
    ba_refinement = cfg.get("ba_refinement", True)
    target_fps = cfg.get("fps", 0)  # 0 means use all frames
    
    if input_path is None:
        print("Error: input_path is required")
        return
        
    if model_path is None:
        print("Using default model path")
        model_path = Path(__file__).parent.parent.parent / "outputs"
    else:
        model_path = Path(model_path)
    
    # Load input data
    if os.path.isdir(input_path):
        print(f"Loading frames from directory: {input_path}")
        frames, _ = load_frames(input_path)
    else:
        print(f"Loading video from: {input_path}")
        frames, fps = load_video(input_path)
    
    if not frames:
        print("Error: No frames loaded")
        return
        
    print(f"Loaded {len(frames)} frames")
    
    # Subsample frames if target_fps is specified
    if target_fps > 0 and fps:
        frames = subsample_frames(frames, original_fps=fps, target_fps=target_fps)
        print(f"Subsampled frames to {len(frames)} frames at {target_fps} fps")
    
    # Format frames for processing
    frames = format_frames(frames, target_size=image_size)
    print(f"Resized frames to {frames[0].shape[:2]}")
    
    # Load model
    print(f"Loading model from {model_path}")
    model, criterion = load_anycam(model_path, checkpoint)
    model = model.cuda().eval()
    
    # Process frames
    trajectory, proj, extras_dict, ba_extras = process_video(
        model, 
        criterion, 
        frames, 
        ba_refinement=ba_refinement
    )

    trajectory = [se3_ensure_numerical_accuracy(torch.tensor(pose)) for pose in trajectory]
    
    # Extract depth and uncertainty information
    best_candidate = extras_dict["best_candidate"]
    depths = extras_dict["seq_depths"]

    if not ba_refinement:
        read_frames = frames
        frames = extras_dict["images"].permute(0, 2, 3, 1).cpu().numpy()
        keyframes = [i for i in range(len(trajectory))]
        uncertainties = torch.stack(extras_dict["uncertainties"])[:, 0, best_candidate, :1, :, :]
    else:
        keyframes = [i * 3 for i in range(len(trajectory) // 3)]
        uncertainties = extras_dict["ba_uncertainties"]
        read_frames = frames


    uncertainties = torch.cat((uncertainties, uncertainties[-1:]), dim=0)
    
    
    print(f"Processed video: {len(trajectory)} poses, {len(depths)} depth maps")
    
    if export_colmap:
        if output_path is None:
            print("Warning: output_path not specified, using temporary directory")
        
        from anycam.utils.colmap_io import export_to_colmap
        
        print("Exporting results to COLMAP format...")
        colmap_path = export_to_colmap(
            trajectory=trajectory,
            proj=proj,
            imgs=read_frames,
            out_dir=output_path
        )
        print(f"Exported COLMAP reconstruction to {colmap_path}")
    
    # Save trajectory and projection matrix if output_path is specified
    if output_path and not export_colmap:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving results to {output_path}")
        
        # Save trajectory as numpy array
        trajectory_np = np.stack([pose.cpu().numpy() for pose in trajectory])
        np.save(output_path / "trajectory.npy", trajectory_np)
        
        # Save projection matrix
        np.save(output_path / "projection.npy", proj.cpu().numpy())
        
        # Save depths if available
        if depths:
            depths_np = np.stack([depth.cpu().numpy() for depth in depths])
            np.save(output_path / "depths.npy", depths_np)
        
        # Save uncertainties if available
        if uncertainties is not None:
            uncertainties_np = np.stack([uncert.cpu().numpy() for uncert in uncertainties])
            np.save(output_path / "uncertainties.npy", uncertainties_np)
            
        print("Saved all results successfully")

        # Visualization or export
    if visualize:
        # Get visualization parameters from vis subconfig
        vis_config = cfg.get("vis", {})
        
        print(f"Visualizing results with rerun (mode: {rerun_mode})...")
        plot_to_rerun(
            trajectory=trajectory,
            depths=depths,
            imgs=frames,
            proj=proj,
            uncertainties=uncertainties,
            subsample_pts=vis_config.get("subsample_pts", 2),
            radii=vis_config.get("radii", 1.5),
            uncertainty_thresh=vis_config.get("uncertainty_thresh", 0.05),
            max_depth=vis_config.get("max_depth", -1),
            filter_depth_threshold=vis_config.get("filter_depth_threshold", 0.1),
            image_plane_distance=vis_config.get("image_plane_distance", 0.05),
            keyframes=keyframes,
            rerun_mode=rerun_mode,
            rerun_address=rerun_address
        )
        
    print("Done")


if __name__ == "__main__":
    main()
