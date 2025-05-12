import argparse
import uuid
from hydra import compose, initialize
import hydra
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.utils import flow_to_image

import torch
from tqdm import tqdm
import sys
import os

sys.path.append(".")
os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))

from anycam.common.array_operations import to
from anycam.visualization.common import color_tensor

from anycam.datasets import make_datasets
from anycam.utils.colmap_io import get_poses_from_colmap
from anycam.utils.droidslam_io import get_poses_from_droidslam
from anycam.loss.metric import translation_angle
from anycam.scripts.common import get_checkpoint_path, load_model
from anycam.loss import make_loss
from anycam.utils.geometry import se3_ensure_numerical_accuracy

from minipytorch3d.rotation_conversions import matrix_to_quaternion, quaternion_to_matrix


from evo.core import metrics
from evo.core.trajectory import PosePath3D
from evo.core import lie_algebra as lie

import rerun as rr
import time
import yaml
import tabulate

import logging
logger = logging.getLogger(__name__)


def ndarray_representer(dumper: yaml.Dumper, array: np.ndarray) -> yaml.Node:
    return dumper.represent_list(array.tolist())

yaml.add_representer(np.ndarray, ndarray_representer)


class PoseAccumulator:
    def __init__(self, model, mode="incremental"):
        self.poses = [np.eye(4)]
        self.gt_poses = [np.eye(4)]
        self.gt_proj = None
        self.imgs = []

        self.model = model
        self.mode = mode

    def update(self, data):
        gt_pose0to1 = torch.inverse(data["poses"][0, 1].to(torch.float64)) @ data["poses"][0, 0].to(torch.float64)
        self.gt_poses.append(self.gt_poses[-1] @ gt_pose0to1.inverse().cpu().numpy())

        if self.mode == "incremental":
            with torch.autocast(device_type="cuda"):
                data = self.model(data)

            pred_pose0to1 = data["proc_poses"][0, 0].to(torch.float64)
            self.poses.append(self.poses[-1] @ pred_pose0to1.inverse().cpu().numpy())
        else:
            img = (data["imgs"][0][0].cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
            self.imgs.append(img)

        self.gt_proj = data["projs"][0, 0].cpu().numpy()

        return data

    def get_trajectory(self, seq_name):
        if self.mode == "global":
            self.poses, self.proj = self.model(self.imgs, self.gt_proj, seq_name)
        else:
            self.poses = self.poses[:-1]

        self.gt_poses = self.gt_poses[:-1]

        return self.poses, self.gt_poses, self.proj, self.gt_proj


class RTE(metrics.RPE):
    @staticmethod
    def rpe_base(Q_i: np.ndarray, Q_i_delta: np.ndarray, P_i: np.ndarray,
                 P_i_delta: np.ndarray) -> np.ndarray:
        """
        Computes the relative SE(3) error pose for a single pose pair
        following the notation of the TUM RGB-D paper.
        :param Q_i: reference SE(3) pose at i
        :param Q_i_delta: reference SE(3) pose at i+delta
        :param P_i: estimated SE(3) pose at i
        :param P_i_delta: estimated SE(3) pose at i+delta
        :return: the RPE matrix E_i in SE(3)
        """
        Q_rel = lie.relative_se3(Q_i, Q_i_delta)
        P_rel = lie.relative_se3(P_i, P_i_delta)
        t_Q = Q_rel[:3, 3]
        t_P = P_rel[:3, 3]

        E = translation_angle(torch.tensor(t_Q[None, ]), torch.tensor(t_P[None, ])).numpy()
        E = np.array([[0, 0, 0, E[0]]])
        return E 


def get_anycam_function(model_path, fit_video_config=None):
    from anycam.scripts.fit_video import fit_video, fit_video_wrapper
    import omegaconf
    from dotdict import dotdict

    model_path = Path(model_path)

    if not model_path.is_dir():
        config_path = model_path.parent
    else:
        config_path = model_path
    name = config_path.name

    training_config = omegaconf.OmegaConf.load(config_path / "training_config.yaml")
    training_config["model"]["use_provided_flow"] = False

    checkpoint_path = get_checkpoint_path(model_path)

    logger.info(f"Loading model from {checkpoint_path}")

    model = load_model(training_config, checkpoint_path,  {"pose_predictor": {"rotation_parameterization": "axis-angle"}}).to("cuda")

    criterion = [make_loss(cfg) for cfg in training_config.get("loss", [])][0]

    fit_video_default_config = OmegaConf.create({
        "with_rerun": False,
        "overfit": False,
        "pyramid": [0],
        "do_ba_refinement": False,
    })

    if fit_video_config is not None:
        fit_video_config = omegaconf.OmegaConf.merge(fit_video_default_config, fit_video_config)

    def run_anycam(imgs, proj, seq_name):
        imgs = [img.astype(np.float32) / 255 for img in imgs]
        poses, projs = fit_video_wrapper(fit_video_config, model, criterion, imgs, 'cuda', gt_proj=proj)
        return poses, projs
    return run_anycam


def get_vggsfm_function():
    colmap_command_template = "cd /storage/user/wimbauer/bts_2/vggsfm; /usr/bin/env /usr/wiss/wimbauer/miniconda3/envs/vggsfm_tmp/bin/python demo.py SCENE_DIR={} shared_camera=True"
    out_dir = f"/storage/user/wimbauer/bts_2/vggsfm/results/{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    
    def run_vggsfm(imgs, proj, seq_name):
        poses, projs, colmap_cli_output  = get_poses_from_colmap(imgs, colmap_command_template, seq_name, out_dir=out_dir)
        if poses is not None:
            to_base_pose = np.linalg.inv(poses[0])
            poses = [to_base_pose @ pose for pose in poses]
        return poses, projs
    
    return run_vggsfm


def get_vggsfm_video_function():
    colmap_command_template = "cd /storage/user/wimbauer/bts_2/vggsfm; /usr/bin/env /usr/wiss/wimbauer/miniconda3/envs/vggsfm_tmp/bin/python video_demo.py SCENE_DIR={} ++window_size=24 ++camera_type=SIMPLE_PINHOLE"
    out_dir = f"/storage/user/wimbauer/bts_2/vggsfm/results-video/{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    
    def run_vggsfm_video(imgs, proj, seq_name):
        poses, projs, colmap_cli_output  = get_poses_from_colmap(imgs, colmap_command_template, seq_name, out_dir=out_dir)
        if poses is not None:
            to_base_pose = np.linalg.inv(poses[0])
            poses = [to_base_pose @ pose for pose in poses]
        return poses, projs
    
    return run_vggsfm_video


def get_flowmap_function():
    flowmap_command_template = "cd /storage/user/wimbauer/bts_2/flowmap; /usr/bin/env /usr/wiss/wimbauer/miniconda3/envs/flowmap/bin/python -m flowmap.overfit dataset=images dataset.images.root={}/images"
    out_dir = "/storage/user/wimbauer/bts_2/flowmap/results"

    def run_flowmap(imgs, proj, seq_name):
        poses, projs, flowmap_cli_output = get_poses_from_colmap(imgs, flowmap_command_template, seq_name, out_dir=out_dir)
        if poses is not None:
            to_base_pose = np.linalg.inv(poses[0])
            poses = [to_base_pose @ pose for pose in poses]
        return poses, projs
    
    return run_flowmap


def get_droidslam_function():
    droidslam_command_template = "cd /storage/user/wimbauer/bts_2/DROID-SLAM; /usr/bin/env /usr/wiss/wimbauer/miniconda3/envs/droidenv/bin/python demo.py --imagedir={imagedir} --calib={calib} --reconstruction_path={reconstruction_path} --disable_vis --stride=1"
    out_dir = f"/storage/user/wimbauer/bts_2/DROID-SLAM/results/{time.strftime('%Y-%m-%d_%H-%M-%S')}"

    def run_droidslam(imgs, proj, seq_name):
        poses, projs, droidslam_cli_output = get_poses_from_droidslam(imgs, proj, droidslam_command_template, seq_name, out_dir=out_dir)
        if poses is not None:
            to_base_pose = np.linalg.inv(poses[0])
            poses = [to_base_pose @ pose for pose in poses]
        return poses, projs[0]
    
    return run_droidslam


def get_dataset(config: OmegaConf, image_size=None):
    _, test_dataset = make_datasets(config, sequential=True, frame_count=2, return_depth=False, image_size=image_size)
    return test_dataset


def eval_trajectories(pred_traj: list, gt_traj: list, pred_proj, gt_proj, with_rerun=False, curr_rr_frame=0, rerun_log=(), rerun_cam_data=(100, 100, 100)):
    pred_traj = np.stack(pred_traj).astype(np.float64)
    gt_traj = np.stack(gt_traj).astype(np.float64)

    pred_traj = PosePath3D(poses_se3=pred_traj)
    gt_traj = PosePath3D(poses_se3=gt_traj)

    ape = metrics.APE()
    rre = metrics.RPE(pose_relation=metrics.PoseRelation.rotation_angle_deg)
    # rte = RTE(pose_relation=metrics.PoseRelation.translation_part) # This computes the relative angle between the translations rather than the translation error
    rte = metrics.RPE(pose_relation=metrics.PoseRelation.translation_part)

    rre_all = metrics.RPE(pose_relation=metrics.PoseRelation.rotation_angle_deg, all_pairs=True)
    rte_all = RTE(pose_relation=metrics.PoseRelation.translation_part, all_pairs=True)

    try:
        # pred_traj.align(gt_traj, correct_only_scale=True)
        pred_traj.align(gt_traj, correct_scale=True)
        #gt_traj.align(pred_traj, correct_only_scale=True)
    except:
        logger.warning("Could not align trajectories")

    ape.process_data((pred_traj, gt_traj))
    rre.process_data((pred_traj, gt_traj))
    rte.process_data((pred_traj, gt_traj))
    rre_all.process_data((pred_traj, gt_traj))
    rte_all.process_data((pred_traj, gt_traj))

    ape_errors = ape.error
    rre_errors = rre.error
    rte_errors = rte.error

    all_rre_errors = rre_all.error
    all_rte_errors = rte_all.error

    ape_result = float(ape_errors.mean())
    rre_result = float(rre_errors.mean())
    rte_result = float(rte_errors.mean())

    auc_05 = (0.5 - np.minimum(rre_errors, rte_errors).clip(0, 0.5)).mean() / 0.5
    auc_1 = (1 - np.minimum(rre_errors, rte_errors).clip(0, 1)).mean()
    auc_3 = (3 - np.minimum(rre_errors, rte_errors).clip(0, 3)).mean() / 3
    auc_10 = (10 - np.minimum(rre_errors, rte_errors).clip(0, 10)).mean() / 10
    rre_0_01 = (rre_errors < 0.01).mean()
    rte_0_01 = (rte_errors < 0.01).mean()

    rre_0_1 = (rre_errors < 0.1).mean()
    rte_0_1 = (rte_errors < 0.1).mean()

    rre_1 = (rre_errors < 1).mean()
    rte_1 = (rte_errors < 1).mean()

    rre_5 = (rre_errors < 5).mean()
    rte_5 = (rte_errors < 5).mean()

    all_auc_3 = (3 - np.minimum(all_rre_errors, all_rte_errors).clip(0, 3)).mean() / 3
    all_auc_5 = (5 - np.minimum(all_rre_errors, all_rte_errors).clip(0, 5)).mean() / 5
    all_auc_10 = (10 - np.minimum(all_rre_errors, all_rte_errors).clip(0, 10)).mean() / 10
    all_auc_30 = (30 - np.minimum(all_rre_errors, all_rte_errors).clip(0, 30)).mean() / 30

    all_rre_1 = (all_rre_errors < 1).mean()
    all_rte_1 = (all_rte_errors < 1).mean()
    all_rre_5 = (all_rre_errors < 5).mean()
    all_rte_5 = (all_rte_errors < 5).mean()

    all_rre_15 = (all_rre_errors < 15).mean()
    all_rte_15 = (all_rte_errors < 15).mean()

    pred_fx = pred_proj[0, 0].item()
    pred_fy = pred_proj[1, 1].item()
    pred_cx = pred_proj[0, 2].item()
    pred_cy = pred_proj[1, 2].item()

    gt_fx = gt_proj[0, 0].item()
    gt_fy = gt_proj[1, 1].item()
    gt_cx = gt_proj[0, 2].item()
    gt_cy = gt_proj[1, 2].item()

    print(pred_fx, pred_fy, gt_fx, gt_fy)

    mean_fx_error = np.abs(pred_fx - gt_fx)
    mean_fy_error = np.abs(pred_fy - gt_fy)

    mean_cx_error = np.abs(pred_cx - gt_cx)
    mean_cy_error = np.abs(pred_cy - gt_cy)

    fx_below_10 = np.abs(pred_fx - gt_fx) < 10
    fy_below_10 = np.abs(pred_fy - gt_fy) < 10

    fx_below_40 = np.abs(pred_fx - gt_fx) < 40
    fy_below_40 = np.abs(pred_fy - gt_fy) < 40

    rel_fx_error = np.abs(pred_fx - gt_fx) / gt_fx
    rel_fy_error = np.abs(pred_fy - gt_fy) / gt_fy

    ape_errors = ape_errors.tolist()
    rre_errors = rre_errors.tolist()
    rte_errors = rte_errors.tolist()

    all_rre_errors = all_rre_errors.tolist()
    all_rte_errors = all_rte_errors.tolist()

    result = {
        "ape_mean": ape_result, 
        "rre_mean": rre_result, 
        "rte_mean": rte_result,
        "ape_errors": ape_errors,
        "rre_errors": rre_errors,
        "rte_errors": rte_errors,
        "all_rre_errors": all_rre_errors,
        "all_rte_errors": all_rte_errors,
        "auc_05": float(auc_05),
        "auc_1": float(auc_1),
        "auc_3": float(auc_3),
        "auc_10": float(auc_10),
        "rre_0_01": float(rre_0_01),
        "rte_0_01": float(rte_0_01),
        "rre_0_1": float(rre_0_1),
        "rte_0_1": float(rte_0_1),
        "rre_1": float(rre_1),
        "rte_1": float(rte_1),
        "rre_5": float(rre_5),
        "rte_5": float(rte_5),
        "all_auc_3": float(all_auc_3),
        "all_auc_5": float(all_auc_5),
        "all_auc_10": float(all_auc_10),
        "all_auc_30": float(all_auc_30),
        "all_rre_1": float(all_rre_1),
        "all_rte_1": float(all_rte_1),
        "all_rre_5": float(all_rre_5),
        "all_rte_5": float(all_rte_5),
        "all_rre_15": float(all_rre_15),
        "all_rte_15": float(all_rte_15),
        "mean_fx_error": float(mean_fx_error),
        "mean_fy_error": float(mean_fy_error),
        "mean_cx_error": float(mean_cx_error),
        "mean_cy_error": float(mean_cy_error),
        "fx_below_10": float(fx_below_10),
        "fy_below_10": float(fy_below_10),
        "fx_below_40": float(fx_below_40),
        "fy_below_40": float(fy_below_40),
        "rel_fx_error": float(rel_fx_error),
        "rel_fy_error": float(rel_fy_error),
        "traj_len": pred_traj.num_poses,
        }

    if with_rerun and "pose" in rerun_log:
        rr.set_time_sequence("frame_idx", curr_rr_frame)
        # rr.log("world", rr.Clear(recursive=True))

        gt_poses = np.array(gt_traj.poses_se3).astype(np.float32)
        pred_poses = np.array(pred_traj.poses_se3).astype(np.float32)

        for i in range(pred_traj.num_poses):
            rr.set_time_sequence("frame_idx", i+curr_rr_frame)
            rr.log(
                "world/gt/cam",
                rr.Pinhole(
                    resolution=rerun_cam_data[:2],
                    focal_length=rerun_cam_data[2],
                )
            )
            rr.log(
                "world/pred/cam",
                rr.Pinhole(
                    resolution=rerun_cam_data[:2],
                    focal_length=rerun_cam_data[2],
                )
            )
            rr.log("world/gt/cam", rr.Transform3D(translation=gt_poses[i][:3, 3], mat3x3=gt_poses[i][:3, :3]))
            rr.log("world/pred/cam", rr.Transform3D(translation=pred_poses[i][:3, 3], mat3x3=pred_poses[i][:3, :3]))

            rr.log("world/gt/traj", rr.LineStrips3D([list(gt_poses[:i+1, :3, 3])], colors=[[255, 0, 0]]))
            rr.log("world/pred/traj", rr.LineStrips3D([list(pred_poses[:i+1, :3, 3])], colors=[[0, 255, 0]]))

    return result


@torch.no_grad()
def run_eval(model, dataloader, with_rerun=False, rerun_log=[], mode="incremental", device="cuda", stop=-1, max_frames=-1):
    try:
        model.eval() 
    except:
        pass

    curr_sequence = None

    proc_traj = [torch.eye(4, device=device, dtype=torch.float64)]
    gt_traj = [torch.eye(4, device=device, dtype=torch.float64)]

    pose_accumulator = PoseAccumulator(model, mode=mode)

    cam_data = (100, 100, 100)

    results = {}
    failed_sequences = []

    curr_rr_frame = 0
    curr_seq_len = 0

    stop = -1
    max_frames = -1

    pbar = tqdm(dataloader)

    for i, data in enumerate(pbar):
        index = data["data_id"].item()

        sequence = dataloader.dataset.get_sequence(index)

        curr_seq_len += 1

        if curr_sequence is None:
            curr_sequence = sequence
        elif curr_sequence != sequence or (stop > 0 and i > stop):
            proc_traj, gt_traj, pred_proj, gt_proj = pose_accumulator.get_trajectory(curr_sequence)

            proc_traj = [se3_ensure_numerical_accuracy(pose) for pose in proc_traj]

            if proc_traj is None:
                print(f"Failure to reconstruct trajectory for sequence {curr_sequence}")
                failed_sequences.append(curr_sequence)
            else:

                seq_result = eval_trajectories(proc_traj, gt_traj, pred_proj, gt_proj, with_rerun, curr_rr_frame, rerun_log, rerun_cam_data=cam_data)
                results[curr_sequence] = seq_result

                print("Results: ", curr_sequence)
                print(tabulate.tabulate([[k, v] for k, v in seq_result.items() if not k.endswith("errors")]))
                print("")

            curr_rr_frame += len(gt_traj)

            pose_accumulator = PoseAccumulator(model, mode=mode)

            curr_sequence = sequence

            curr_seq_len = 1

        if max_frames > 0 and curr_seq_len > max_frames:
            continue

        data = to(data, device)

        data = pose_accumulator.update(data)

        if with_rerun:
            rr.set_time_sequence("frame_idx", curr_rr_frame + curr_seq_len - 1)
            h, w = data["imgs"].shape[-2:]

            if "img" in rerun_log:
                rr.log("world/pred/img", rr.Image((data["imgs"][0, 0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).compress(jpeg_quality=95))
            if "gt_flow" in rerun_log:
                flow = data["images_ip"][:, 0, 3:5].permute(0, 2, 3, 1)
                flow = torch.cat((flow[:, :, :, 0:1] / 2 * w , flow[:, :, :, 1:2] / 2 * h), dim=-1).permute(0, 3, 1, 2)
                flow_img = flow_to_image(flow[0].cpu()).numpy().astype(np.uint8).transpose(1, 2, 0)
                rr.log("world/gt/flow", rr.Image(flow_img).compress(jpeg_quality=95))
            if "induced_flow" in rerun_log:
                induced_flow = data["induced_flow"][:, 0].permute(0, 2, 3, 1).clamp(-1, 1)
                induced_flow = torch.cat((induced_flow[:, :, :, 0:1] / 2 * w , induced_flow[:, :, :, 1:2] / 2 * h), dim=-1).permute(0, 3, 1, 2)
                induced_flow_img = flow_to_image(induced_flow[0].cpu()).numpy().astype(np.uint8).transpose(1, 2, 0)
                rr.log("world/pred/flow", rr.Image(induced_flow_img).compress(jpeg_quality=95))
            if "flow_diff" in rerun_log and "gt_flow" in rerun_log and "induced_flow" in rerun_log:
                flow_diff = ((flow - induced_flow).norm(dim=1) / 10).clamp(0, 1)
                flow_diff_img = (color_tensor(flow_diff, cmap="turbo")[0].cpu().numpy() * 255).astype(np.uint8)
                rr.log("world/flow_diff", rr.Image(flow_diff_img).compress(jpeg_quality=95))
            if "depth" in rerun_log:
                depth = data["scaled_depths"][0][0, 0].cpu().squeeze().numpy()
                rr.log("world/pred/cam/depth", rr.DepthImage(depth, meter=1))
            
            cam_data = (w, h, data["projs"][0, 0, 0, 0].item())

        if (stop > 0 and i > stop):
            break

    if len(pose_accumulator.imgs) > 1:
        proc_traj, gt_traj, pred_proj, gt_proj = pose_accumulator.get_trajectory(curr_sequence)

        proc_traj = [se3_ensure_numerical_accuracy(pose) for pose in proc_traj]

        if proc_traj is None:
            print(f"Failure to reconstruct trajectory for sequence {curr_sequence}")
            failed_sequences.append(curr_sequence)
        else:
            seq_result = eval_trajectories(proc_traj, gt_traj, pred_proj, gt_proj, with_rerun, curr_rr_frame, rerun_log, rerun_cam_data=cam_data)
            results[curr_sequence] = seq_result

        print("Results: ", curr_sequence)
        print(tabulate.tabulate([[k, v] for k, v in seq_result.items() if not k.endswith("errors")]))
        print("")
    else:
        print("No data for sequence: ", curr_sequence)
        curr_sequence = list(results.keys())[-1]

    avg_results = {}
    for key in results[curr_sequence].keys():
        if key.endswith("errors"):
            continue
        avg_results[key] = float(np.mean([res[key] for res in results.values()]))
        
    print("Avg Results: ")
    print(tabulate.tabulate([[k, v] for k, v in avg_results.items() if not k.endswith("errors")]))
    print("")

    print("Failed sequences: ", failed_sequences)
    print("")

    return avg_results, results, failed_sequences


@hydra.main(version_base=None, config_path="../../anycam/configs")
def main(conf):
    model_path = Path(conf.model_path)
    out_path = Path(conf.out_path)
    other_model = conf.other_model
    with_rerun = conf.with_rerun
    rerun_log = conf.rerun_log
    save_rerun = conf.save_rerun
    fit_video_config = conf.fit_video
    image_size = conf.image_size

    if type(image_size) == int and image_size <= 0:
        image_size = None

    if other_model is None:
        if not model_path.is_dir():
            config_path = model_path.parent
        else:
            config_path = model_path
        name = config_path.name

        config = OmegaConf.load(config_path / "training_config.yaml")

        checkpoint_path = get_checkpoint_path(model_path)

        config["model"]["use_provided_flow"] = False
        model = load_model(config, checkpoint_path).to("cuda")
        mode = "incremental"
    else:
        if other_model == "vggsfm":
            model = get_vggsfm_function()
            name = "vggsfm"
            mode = "global"
        elif other_model == "vggsfm-video":
            model = get_vggsfm_video_function()
            name = "vggsfm-video"
            mode = "global"
        elif other_model == "flowmap":
            model = get_flowmap_function()
            name = "flowmap"
            mode = "global"
        elif other_model == "droidslam":
            model = get_droidslam_function()
            name = "droidslam"
            mode = "global"
        elif other_model == "anycam":
            model = get_anycam_function(model_path, fit_video_config)
            name = "anycam"
            mode = "global"
        elif other_model == "dust3r":
            raise NotImplementedError
        elif other_model == "relpose++":
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown model: {other_model}")
   
    dataset_path = Path(conf.dataset)

    dataset_config = OmegaConf.load(dataset_path)

    # Set data root
    dataset_config.data_root = conf.data_root
    
    dataset = get_dataset(dataset_config, image_size=image_size)
    dataset._datapoints = dataset._datapoints
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)

    out_path = out_path / f"{name}_{dataset.NAME}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"

    try:
        model.eval() 
    except:
        pass

    if with_rerun:
        rr.init("trajectory evaluation", recording_id=uuid.uuid4())
        if save_rerun:
            rr.save(out_path / "estimated_trajectories.rrd")
        else:
            rr.connect()
        
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    avg_results, results, failed_sequences = run_eval(model, dataloader, with_rerun, rerun_log, mode=mode)

    out_path.mkdir(parents=True, exist_ok=True)

    with open(out_path / "results.yaml", "w") as f:
        yaml.dump(results, f)

    with open(out_path / "avg_results.yaml", "w") as f:
        yaml.dump(avg_results, f)

    with open(out_path / "failed_sequences.yaml", "w") as f:
        yaml.dump(failed_sequences, f)

    info = {
        "model_path": str(model_path),
        "config_path": str(config_path) if other_model is None else None,
        "dataset_path": str(dataset_path),
        "split_path": conf.split_path,
        "other_model": other_model,
        "with_rerun": with_rerun,
    }

    with open(out_path / "info.yaml", "w") as f:
        yaml.dump(info, f)

    print("Done")


if __name__ == "__main__":
    main()
