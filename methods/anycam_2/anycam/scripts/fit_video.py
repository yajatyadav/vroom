import argparse
from copy import deepcopy
import copy
import logging
from pathlib import Path
import uuid

import cv2
import hydra
from hydra import initialize, compose
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from omegaconf import DictConfig
import omegaconf
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.utils import flow_to_image

import sys


sys.path.append("../..")
sys.path.append(".")

from anycam.visualization.common import color_tensor
from anycam.loss import make_loss
from anycam.scripts.common import get_checkpoint_path, load_model
from anycam.trainer import induce_flow_dist, make_proj_from_focal_length
from anycam.utils.geometry import average_pose
from anycam.utils.bundle_adjustment import *
from anycam.loss.metric import rotation_angle

try:
    import rerun as rr
except:
    rr = None


logger = logging.getLogger(__name__)


def load_images(input_path):
    img_paths = sorted(list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")) + list(input_path.glob("*.jpeg")))

    imgs = []

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255
        imgs.append(img)

    imgs = np.stack(imgs)

    return imgs


def make_dataset(config, imgs, device):

    target_size = config.get("image_size", None)
    center_crop = config.get("center_crop", False)

    class ImageDataset(Dataset):
        def __init__(self, imgs, target_size=None, center_crop=False):
            self.imgs = torch.tensor(np.array(imgs), device=device).permute(0, 3, 1, 2)

            bh, bw = self.imgs.shape[-2:]

            if target_size is not None:
                h, w = target_size

                if h is None:
                    h = int(bh / bw * w)
                elif w is None:
                    w = int(bw / bh * h)

                self.imgs = F.interpolate(self.imgs, (h, w), mode="bilinear", align_corners=False)
            else:
                h, w = self.imgs.shape[-2:]

            self.scale_factor = h / bh

            if center_crop:
                h, w = self.imgs.shape[-2:]

                h_ = min(h, w)
                w_ = min(h, w)

                h_start = (h - h_) // 2
                w_start = (w - w_) // 2

                self.imgs = self.imgs[:, :, h_start:h_start+h_, w_start:w_start+w_]

        def __len__(self):
            return len(self.imgs)
        
        def __getitem__(self, idx):
            img = self.imgs[idx]
            return img

    return ImageDataset(imgs, target_size=target_size, center_crop=center_crop)


def fit_video_wrapper(config, model, criterion, imgs, device, gt_proj=None):
    pyramid = config.get("pyramid", 1)

    if type(pyramid) == int:
        pyramid = [pyramid]

    candidates = []

    # Use gt focal or not
    if not config.get("use_gt_focal", False):
        gt_proj = None

    print(f"Using GT focal: {config.get('use_gt_focal', False)} {gt_proj is not None}")

    for level in pyramid:
        imgs_level = imgs[::(level + 1)]

        level_poses, level_proj = fit_video(config, model, criterion, imgs_level, device=device, gt_proj=gt_proj)

        interpolated_poses = []
        interpolated_proj = []

        for i in range(len(imgs)):
            if i % (level + 1) == 0:
                interpolated_poses.append(torch.tensor(level_poses[i // (level + 1)]))
            else:
                prev_pose = level_poses[i // (level + 1)]
                if i // (level + 1) + 1 < len(level_poses):
                    next_pose = level_poses[(i // (level + 1)) + 1]

                    t = (i % (level + 1)) / (level + 1)

                    interpolated_pose = average_pose(torch.stack([torch.tensor(prev_pose), torch.tensor(next_pose)]), weight=t)

                    interpolated_poses.append(interpolated_pose)
                else:
                    interpolated_poses.append(torch.tensor(prev_pose))

            interpolated_proj.append(torch.tensor(level_proj))

        interpolated_poses = torch.stack(interpolated_poses).cpu()
        interpolated_proj = torch.stack(interpolated_proj).cpu()

        candidates.append(interpolated_poses)

    final_poses = None
    final_projs = interpolated_proj.mean(dim=0)

    for i, poses in enumerate(candidates):
        if final_poses is None:
            final_poses = poses
        else:
            final_poses = average_pose(torch.stack([final_poses, poses]), weight=1/(i+1))

    return final_poses, final_projs


@torch.no_grad()
def compute_depth_flow(model, imgs=None, imgs0=None, imgs1=None):

    seq_imgs = []
    seq_depths = []
    seq_flow_occs_fwd = []
    seq_flow_occs_bwd = []

    if imgs is not None:
        imgs0 = imgs[:-1]
        imgs1 = imgs[1:]
    else:
        assert imgs0 is not None
        assert imgs1 is not None

    for (i, (img0, img1)) in tqdm(list(enumerate(zip(imgs0, imgs1)))):
        img_pair = torch.stack([img0, img1]).unsqueeze(0).cuda()

        images_ip_fwd, images_ip_bwd = model.image_processor(img_pair * 2 - 1, data={})

        depth = model.depth_predictor(img0.unsqueeze(0).cuda())

        depth = 1 / depth[0].clamp_min(1e-3)

        seq_imgs.append(img0.cpu())

        if imgs is not None:
            seq_flow_occs_fwd.append(images_ip_fwd[0, :(1 if i != len(imgs0)-1 else 2), 3:6].cpu())
            seq_flow_occs_bwd.append(images_ip_bwd[0, (1 if i != 0 else 0):, 3:6].cpu())
        else:
            seq_flow_occs_fwd.append(images_ip_fwd[0, :1, 3:6].cpu())
            seq_flow_occs_bwd.append(images_ip_bwd[0, 1:, 3:6].cpu())

        seq_depths.append(depth.cpu())

    if imgs is not None:
        depth = model.depth_predictor(img1.unsqueeze(0).cuda())
        depth = 1 / depth[0].clamp_min(1e-3)

        seq_imgs.append(img1.cpu())
        seq_depths.append(depth.cpu())

    seq_imgs = torch.stack(seq_imgs, dim=0)
    seq_depths = torch.cat(seq_depths, dim=0)
    seq_flow_occs_fwd = torch.cat(seq_flow_occs_fwd, dim=0)
    seq_flow_occs_bwd = torch.cat(seq_flow_occs_bwd, dim=0)

    return seq_imgs, seq_depths, seq_flow_occs_fwd, seq_flow_occs_bwd


@torch.autocast(device_type="cuda", enabled=True)
@torch.no_grad()
def fit_video(config, model, criterion, imgs, device="cuda", return_extras=False, gt_proj=None):

    print(config)

    dataset_config = config.get("dataset", {})

    do_ba_refinement = config.get("do_ba_refinement", False)
    ba_refinement_level = config.get("ba_refinement_level", 0) + 1
    ba_refinement_config = config.get("ba_refinement", {})

    prediction_config = config.get("prediction", {})

    model_seq_len = prediction_config.get("model_seq_len", 64)
    shift = prediction_config.get("shift", 63)
    square_crop = prediction_config.get("square_crop", False)
    return_all_uncerts = prediction_config.get("return_all_uncerts", False)

    proj_strategy = prediction_config.get("proj_strategy", "weighted")
    proj_label_source = prediction_config.get("proj_label", "prediction")

    print(f"dataset_config: {dataset_config}")
    print(f"do_ba_refinement: {do_ba_refinement}")
    print(f"ba_refinement_level: {ba_refinement_level}")
    print(f"ba_refinement_config: {ba_refinement_config}")
    print(f"prediction_config: {prediction_config}")
    print(f"model_seq_len: {model_seq_len}")
    print(f"shift: {shift}")
    print(f"proj_strategy: {proj_strategy}")
    print(f"proj_label_source: {proj_label_source}")


    dataset = make_dataset(dataset_config, imgs, device="cpu")

    if config.with_rerun:
        rr.init("Prediction", recording_id=uuid.uuid4())
        rr.connect()

        for i, img in enumerate(dataset.imgs):
            rr.set_time_sequence("timestep", i)
            rr.log(f"world/img", rr.Image((img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)).compress(jpeg_quality=95))

    # Preprocess all images

    logger.info("Preprocessing images")

    dont_compute = False

    if model.pose_predictor.backbone_type == "croco":
        logger.info("Ignore flow and depth for CroCo")
        dont_compute = True



    c, h, w = dataset.imgs.shape[1:]

    if square_crop:
        sq = min(h, w)
        h_, w_ = sq, sq
    else:
        h_, w_ = h, w

    seq_imgs = dataset.imgs

    if square_crop:
        seq_imgs = seq_imgs[:, :, (h-sq)//2:(h-sq)//2+sq, (w-sq)//2:(w-sq)//2+sq]

    seq_imgs, seq_depths, seq_flow_occs_fwd, seq_flow_occs_bwd = compute_depth_flow(model, seq_imgs)
    
    def prepare_batch(batch_ids_ids):
        batch_size, frame_count = batch_ids_ids.shape

        batch_ids_ids = batch_ids_ids.cpu()

        imgs = seq_imgs[batch_ids_ids.view(-1), :, :, :].reshape(batch_size, frame_count, c, h_, w_)
        depths = seq_depths[batch_ids_ids.view(-1)].reshape(batch_size, frame_count, 1, h_, w_)
        flow_occ_fwd = seq_flow_occs_fwd[batch_ids_ids.view(-1)].reshape(batch_size, frame_count, 3, h_, w_)
        flow_occ_bwd = seq_flow_occs_bwd[batch_ids_ids.view(-1)].reshape(batch_size, frame_count, 3, h_, w_)

        # Concatenate forward and backward
        imgs = torch.cat([imgs, imgs.flip(1)], dim=0).cuda()
        depths = torch.cat([depths, depths.flip(1)], dim=0).cuda()
        flow_occs = torch.cat([flow_occ_fwd, flow_occ_bwd.flip(1)], dim=0).cuda()

        return imgs, depths, flow_occs
    
    # device = "cuda"
    
    candidate_trajectories = [torch.eye(4, device=device).view(1, 4, 4).expand(model.pose_predictor.focal_num_candidates, -1, -1)]
    sub_trajectories = []
    
    proj_labels = []

    uncertainties = []
    angle_sum = 0

    pose_predictor = model.pose_predictor
    pose_predictor.eval()

    for i in tqdm(range(0, len(dataset)-1, shift)):
        batch_ids = torch.arange(i, i+1, device=device).view(-1, 1)
        seq_len_ = min(model_seq_len, len(dataset)-i)
        ids = torch.arange(0, seq_len_, device=device).view(1, -1)
        batch_ids_ids = batch_ids + ids

        imgs, depths, flow_occs, = prepare_batch(batch_ids_ids)

        flow_occs[:, -1, :2] = 0
        flow_occs[:, -1, 2] = 1

        pose_result = pose_predictor(
            images=imgs,
            depths=depths,
            flow_occs=flow_occs,
        )

        if proj_label_source == "prediction":
            proj_label = pose_result["focal_length_probs"][:, 0]
        elif proj_label_source == "loss":
            if model.try_focal_length_candidates:
                proj_candidates = make_proj_from_focal_length(pose_result["focal_length_candidates"], w/h)
            else:
                proj_candidates = make_proj_from_focal_length(pose_result["focal_length"].unsqueeze(-1))

            induced_flow, dist = induce_flow_dist(depths.unsqueeze(2), proj_candidates, pose_result["poses"].clone())

            pose_result["flow_occs_in"] = flow_occs
            pose_result["aligned_depths"] = depths.unsqueeze(2)
            pose_result["induced_flow"] = induced_flow
            pose_result["dist"] = dist

            _, losses = criterion({"pose_result": pose_result}, return_extra_data=True)
            proj_label = losses["flow_soft_label"]
        else:
            raise ValueError(f"Unknown proj_label_source: {proj_label_source}")

        num_candidates = proj_label.shape[-1]

        fwd_poses = pose_result["poses"][0, :-1]

        bwd_poses = pose_result["poses"][1, :-1]
        bwd_poses = torch.inverse(bwd_poses.flip(0))

        poses = average_pose(torch.stack([fwd_poses, bwd_poses]))

        with torch.autocast(device_type="cuda", enabled=False):
            r_angles = rotation_angle(poses[:, 16, :3, :3].to(torch.float32), torch.eye(3, device=device).view(1, 3, 3).expand(poses.shape[0], -1, -1))
            mean_angle = torch.mean(r_angles).item()
            mean_angle = 1
            angle_sum = angle_sum + mean_angle

        proj_labels.append(proj_label[0] * mean_angle)

        with torch.autocast(device_type="cuda", dtype=torch.float32):
            sub_trajectory = [torch.eye(4, device=device).view(1, 4, 4).expand(num_candidates, -1, -1)]

            for j in range(seq_len_-1):
                sub_trajectory.append(sub_trajectory[-1] @ torch.inverse(poses[j, :]))

            extra_poses = candidate_trajectories[i:]

            candidate_trajectories = candidate_trajectories[:i+1]
            uncertainties = uncertainties[:i]

            last_pose = candidate_trajectories[-1]

            sub_trajectory = [last_pose @ pose for pose in sub_trajectory]

            for k, pose in enumerate(poses):
                if k+1 < len(extra_poses):
                    cand_rel_pose = torch.inverse(extra_poses[k+1]) @ extra_poses[k]

                    pose = average_pose(torch.stack([pose, cand_rel_pose]))

                candidate_trajectories.append(candidate_trajectories[-1] @ torch.inverse(pose))
                uncertainties.append(pose_result["uncert"][:, k])

        sub_trajectories.append(sub_trajectory)

        if config.with_rerun:

            cmap = plt.get_cmap('hsv')
            cmap_cycle = 16

            for k in range(0, seq_len_):
                rr.set_time_sequence("timestep", i+k)
                
                img = (imgs[0, k].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

                uncert = pose_result["uncert"][0, k:k+1, 16, 0, :, :].detach()
                uncert = color_tensor(uncert, cmap="plasma", norm=True)[0].cpu().numpy()
                uncert = (uncert * 255).astype(np.uint8)

                rr.log("world/img", rr.Image(img).compress(jpeg_quality=95))
                rr.log("world/uncert", rr.Image(uncert).compress(jpeg_quality=95))

            for i in range(num_candidates):
                rr.log(f"world/cand_{i:04d}/traj", rr.LineStrips3D([[p[i][:3, 3].cpu().numpy().tolist() for p in candidate_trajectories]], colors=[[0, 255, 0]]))

                sub_colors = [(np.array(cmap((k % cmap_cycle) / cmap_cycle)) * 255).astype(np.uint8).tolist() for k in range(len(sub_trajectories))]
                st_plot = [[p[i][:3, 3].cpu().numpy().tolist() for p in st] for st in sub_trajectories]

                rr.log(f"world/cand_{i:04d}/sub_traj", rr.LineStrips3D(st_plot, colors=sub_colors))

                rr.log(f"world/cand_{i:04d}/cam/pinhole", rr.Pinhole(
                    resolution=[w, h],
                    focal_length=w,
                ))

                rr.log(f"world/cand_{i:04d}/cam", rr.Transform3D(translation=candidate_trajectories[-1][i, :3, 3].cpu().numpy(), mat3x3=candidate_trajectories[-1][i, :3, :3].cpu().numpy()))

    # logger.warning(f"Using new pred labels")
    proj_labels = torch.stack(proj_labels).sum(dim=0) / angle_sum

    best_candidate = proj_labels.argmax()

    print(f"Best candidate: {best_candidate.item()}")

    if proj_strategy == "best":
        proj = make_proj_from_focal_length(pose_result["focal_length_candidates"][0:1, best_candidate:best_candidate+1], w_/h_)[0]
    elif proj_strategy == "mean":
        avg_label = (torch.arange(0, proj_labels.shape[-1], device=device) * proj_labels).sum(-1)
        print(avg_label)
        lower = torch.floor(avg_label)
        upper = torch.ceil(avg_label)
        interp = avg_label - lower
        fl = pose_result["focal_length_candidates"][0:1, lower.long()] * (1 - interp) + pose_result["focal_length_candidates"][0:1, upper.long()] * interp
        proj = make_proj_from_focal_length(fl.unsqueeze(1), w_/h_)[0]
    elif proj_strategy == "weighted":
        proj = make_proj_from_focal_length(pose_result["focal_length_candidates"][0:1, :], w_/h_)[0]
        proj = proj * proj_labels.view(-1, 1, 1)
        proj = proj.sum(dim=0, keepdim=True)
    else:
        raise ValueError(f"Unknown proj_strategy: {proj_strategy}")

    if gt_proj is not None:
        # Pick candidate that is closest to GT
        gt_normalized_focal = gt_proj[0, 0] / w_ * 2
        candidates = pose_result["focal_length_candidates"][0]
        candidate_distances = torch.abs(candidates - gt_normalized_focal)
        gt_best_candidate = candidate_distances.argmin()

        print(f"Replacing best candidate with GT candidate {best_candidate.item()} -> {gt_best_candidate.item()}")
        print(f"GT focal: {gt_normalized_focal.item()} vs Candidate focal: {proj[0, 0, 0].item()}")

        best_candidate = gt_best_candidate

    proj[:, 0, 0] = proj[:, 0, 0] * w_ / w
    proj[:, 1, 1] = proj[:, 1, 1] * h_ / h

    proj[:, 0, 0] = (proj[:, 0, 0] * 0.5) * w
    proj[:, 1, 1] = (proj[:, 1, 1] * 0.5) * h
    proj[:, 0, 2] = (proj[:, 0, 2] * 0.5 + 0.5) * w
    proj[:, 1, 2] = (proj[:, 1, 2] * 0.5 + 0.5) * h

    proj = proj[0].cpu().numpy()

    best_trajectory = [p[best_candidate].cpu().numpy() for p in candidate_trajectories]

    if do_ba_refinement:
        if ba_refinement_level > 1 or square_crop:

            if square_crop:
                print("Recomputing uncertainties for ba refinement.")

                ba_uncertainties = []

                print("Reverting back to original resolution.")
                h_, w_ = h, w

                if return_all_uncerts:
                    uncert_level = 1
                else:
                    uncert_level = ba_refinement_level


                imgs0 = dataset.imgs[:-1:uncert_level]
                imgs1 = dataset.imgs[1::uncert_level]

                n_new = len(imgs0)

                seq_imgs, seq_depths, seq_flow_occs_fwd, seq_flow_occs_bwd = compute_depth_flow(model, imgs0=imgs0, imgs1=imgs1)

                seq_imgs = torch.cat([seq_imgs, seq_imgs[-1:]], dim=0)
                seq_depths = torch.cat([seq_depths, seq_depths[-1:]], dim=0)
                seq_flow_occs_fwd = torch.cat([seq_flow_occs_fwd, seq_flow_occs_fwd[-1:]], dim=0)
                seq_flow_occs_bwd = torch.cat([seq_flow_occs_bwd, seq_flow_occs_bwd[-1:]], dim=0)

                for i in tqdm(range(0, n_new, shift)):
                    batch_ids = torch.arange(i, i+1, device=device).view(-1, 1)
                    seq_len_ = min(model_seq_len, n_new+1-i)
                    ids = torch.arange(0, seq_len_, device=device).view(1, -1)
                    batch_ids_ids = batch_ids + ids

                    imgs, depths, flow_occs, = prepare_batch(batch_ids_ids)

                    imgs = imgs[:1]
                    depths = depths[:1]
                    flow_occs = flow_occs[:1]

                    flow_occs[:, -1, :2] = 0
                    flow_occs[:, -1, 2] = 1

                    pose_result = pose_predictor(
                        images=imgs,
                        depths=depths,
                        flow_occs=flow_occs,
                    )
                    ba_uncertainties = ba_uncertainties[:i]

                    for k in range(pose_result["poses"].shape[1]):
                        ba_uncertainties.append(pose_result["uncert"][:, k])

                seq_imgs = seq_imgs[:-1]

                ba_uncertainties = ba_uncertainties[:-1]

                if return_all_uncerts:
                    uncertainties = ba_uncertainties
                    ba_uncertainties = uncertainties[::ba_refinement_level]
                    seq_imgs = seq_imgs[::ba_refinement_level]

                print("Recomputing uncertainties done.", len(ba_uncertainties))

                seq_imgs = dataset.imgs[::ba_refinement_level][:len(ba_uncertainties)]
                # print(len(seq_imgs), len(dataset.imgs[::ba_refinement_level]))
            else:
                seq_imgs = dataset.imgs[::ba_refinement_level]
                ba_uncertainties = uncertainties[::ba_refinement_level]


            c, h, w = dataset.imgs.shape[1:]

            seq_imgs, seq_depths, seq_flow_occs_fwd, seq_flow_occs_bwd = compute_depth_flow(model, seq_imgs)

        else:
            seq_imgs = dataset.imgs


        ba_uncertainties = torch.stack(ba_uncertainties)
        ba_uncertainties = ba_uncertainties[:, 0, best_candidate, :1]


        best_trajectory, proj, ba_extras = ba_refinement(
            ba_refinement_config, 
            best_trajectory[::ba_refinement_level][:len(seq_imgs)], 
            proj, 
            # uncertainties[:len(ba_imgs)-1], 
            ba_uncertainties,
            seq_imgs, 
            seq_depths, 
            seq_flow_occs_fwd, 
            seq_flow_occs_bwd, 
            device=device
        )

        interpolated_poses = []

        l = len(best_trajectory)

        for i in range(len(dataset.imgs)):
            if i % ba_refinement_level == 0 and (i // ba_refinement_level )< l:
                interpolated_poses.append(torch.tensor(best_trajectory[i // ba_refinement_level]))
            else:
                if i // ba_refinement_level + 1 < l:
                    prev_pose = best_trajectory[i // ba_refinement_level]
                    
                    next_pose = best_trajectory[(i // ba_refinement_level) + 1]

                    t = (i % ba_refinement_level) / ba_refinement_level

                    interpolated_pose = average_pose(torch.stack([torch.tensor(prev_pose), torch.tensor(next_pose)]), weight=t)

                    interpolated_poses.append(interpolated_pose)
                else:
                    last1 = interpolated_poses[-2]
                    last0 = interpolated_poses[-1]

                    rel = torch.inverse(last1.to(torch.float32)) @ last0

                    next = last0 @ rel

                    interpolated_poses.append(torch.tensor(next))

        interpolated_poses = torch.stack(interpolated_poses).cpu()

        best_trajectory = interpolated_poses
        
    else:
        ba_extras = None
        ba_uncertainties = None

    proj = proj / dataset.scale_factor

    if not return_extras:
        return best_trajectory, proj
    else:
        extras_dict = {
            "uncertainties": uncertainties, 
            "candidate_trajectories": candidate_trajectories, 
            "pred_labels": proj_labels, 
            "flow_labels": proj_labels,
            "images": seq_imgs, 
            "seq_depths": seq_depths, 
            "seq_flow_occs_fwd": seq_flow_occs_fwd, 
            "seq_flow_occs_bwd": seq_flow_occs_bwd, 
            "ba_uncertainties": ba_uncertainties,
            "best_candidate": best_candidate,
            "focal_length_candidates": pose_result["focal_length_candidates"],
        }
        return best_trajectory, proj, extras_dict, ba_extras



@torch.compile(mode="reduce-overhead", fullgraph=True)
def compute_loss(ba_param_inv_depth, ba_param_rot, ba_param_t, ba_param_focal_length, pixel_tracks, ba_indices, uncerts, w, h, loss_mask, max_uncert, rotation_representation="quaternion"):
    n, wc, gs, tl, _ = pixel_tracks.shape

    fl = (ba_param_focal_length * 2).exp()
    fl = fl.clamp(0.1, 10)

    # n, seq_len, 4, 4
    ba_poses_c2w = param_to_pose(ba_param_rot, ba_param_t)

    ba_poses_w2c = torch.inverse(ba_poses_c2w)

    ba_proj, ba_inv_proj = make_normalized_proj(fl, w/h)
    # ba_inv_proj  = torch.inverse(ba_proj)

    ba_proj = ba_proj.unsqueeze(0)
    ba_inv_proj = ba_inv_proj.unsqueeze(0)

    xy = pixel_tracks[:, :, :, 0]
    xyz = torch.cat([xy, torch.ones_like(xy[..., :1])], dim=-1)
    xyz = xyz.view(n, -1, 3).permute(0, 2, 1)

    xyz_cam = ba_inv_proj @ xyz
    xyz_cam = xyz_cam * (1 / ba_param_inv_depth.view(n, 1, -1).clamp_min(1e-4))
    xyz_cam[ba_param_inv_depth.view(n, 1, -1).expand(-1, 3, -1) < 1e-4] = 0

    xyzh_cam = torch.cat([xyz_cam, torch.ones_like(xyz_cam[:, :1])], dim=1)

    anchor_poses_c2w = get_corr_poses(ba_indices[:, :, :, :1], ba_poses_c2w)

    xyzh_world = anchor_poses_c2w @ xyzh_cam.reshape(n, 4, -1).permute(0, 2, 1).reshape(n, -1, 4, 1)

    ref_poses_w2c = get_corr_poses(ba_indices[:, :, :, 1:], ba_poses_w2c)

    xyzh_world_exp = xyzh_world.reshape(n, wc, gs, 1, 4).expand(n, wc, gs, tl-1, 4).reshape(n, -1, 4, 1)

    xyzh_cam = ref_poses_w2c @ xyzh_world_exp

    xyzh_cam = xyzh_cam.reshape(n, -1, 4).permute(0, 2, 1)

    xyz = xyzh_cam[:, :3, :] / xyzh_cam[:, 2:3, :]
    xyz[xyzh_cam[:, 2:3, :].view(n, 1, -1).expand(-1, 3, -1) < 1e-4] = 0

    xy = (ba_proj @ xyz)[:, :2]

    xy = xy.permute(0, 2, 1).view(n, wc, gs, tl-1, 2)

    dist = xy - pixel_tracks[:, :, :, 1:, :]

    # dist = dist.norm(dim=-1)

    dist = dist.abs().mean(dim=-1)

    # repr_loss = dist / (uncerts[:, :, :, 1:, 0] + 1e-4) ** 2
    # repr_loss = dist / (uncerts[:, :, :, 1:, 0] + 1e-4)
    # repr_loss = dist

    repr_loss = (max_uncert - uncerts[:, :, :, 1:, 0]).clamp_min(0) * dist

    loss_mask_filter = loss_mask

    if max_uncert > 0:
        loss_mask_filter = loss_mask & (uncerts[:, :, :, 1:, 0] < max_uncert)

    repr_loss[~loss_mask_filter] = 0

    
    curr_rel_poses = torch.inverse(ba_poses_c2w[:, :-1]) @ ba_poses_c2w[:, 1:]            
    curr_rel_rot, curr_rel_t = pose_to_param(curr_rel_poses, rotation_representation)
    curr_rel_t_abs = ba_poses_c2w[:, 1:, :3, 3] - ba_poses_c2w[:, :-1, :3, 3]
    # ref_rel_rot, ref_rel_t = pose_to_param(rel_poses, rotation_representation)

    # smoothness_loss_rot = (curr_rel_rot[:, 1:] - curr_rel_rot[:, :-1]).abs().mean()
    # smoothness_loss_t = (curr_rel_t[:, 1:] - curr_rel_t[:, :-1]).abs().mean()
    smoothness_loss_t = (curr_rel_t_abs[:, 1:] - curr_rel_t_abs[:, :-1]).abs().mean()

    smoothness_loss = smoothness_loss_t

    return repr_loss, smoothness_loss, ba_proj, xyzh_world


@torch.autocast(device_type="cuda", dtype=torch.float32)
@torch.enable_grad()
def ba_refinement(config, initial_trajectory, proj, uncertainties, seq_imgs, seq_depths, seq_flow_occs_fwd, seq_flow_occs_bwd, device="cuda"):
    with_rerun = config.get("with_rerun", True)
    ba_window = config.get("ba_window", 8) # 8
    overlap = config.get("overlap", 6) # 4
    rotation_representation = config.get("rotation_representation", "quaternion")

    max_uncert = config.get("max_uncert", -1)
    use_best = config.get("use_best", False)

    lambda_smoothness = config.get("lambda_smoothness", 200) # 2000 

    global_every_n = config.get("global_every_n", 2)

    n_steps_sliding = config.get("n_steps_sliding", 400) # 500 # 250
    n_steps_global = config.get("n_steps_global", 100) # 1000 # 100
    n_steps_last_global = config.get("n_steps_last_global", 5000) # 5000
    n_steps_only_focal = config.get("n_steps_only_focal", 0) # 1000

    all_reg_to_zero = config.get("all_reg_to_zero", True)

    track_len = config.get("track_len", 8) # 8
    stride = config.get("stride", 1)
    grid_size = config.get("grid_size", 16) # 16
    long_tracks = config.get("long_tracks", False) # False

    optimize_relatives = config.get("optimize_relatives", False) # False

    lr = config.get("lr", 1e-4)

    log_interval = config.get("log_interval", 200)
    rerun_offset = 10
    
    # print all parameters
    print(f"ba_window: {ba_window}")
    print(f"overlap: {overlap}")
    print(f"rotation_representation: {rotation_representation}")
    print(f"lr: {lr}")
    print(f"lambda_smoothness: {lambda_smoothness}")
    print(f"global_every_n: {global_every_n}")
    print(f"n_steps_sliding: {n_steps_sliding}")
    print(f"n_steps_global: {n_steps_global}")
    print(f"n_steps_last_global: {n_steps_last_global}")
    print(f"all_reg_to_zero: {all_reg_to_zero}")
    print(f"track_len: {track_len}")
    print(f"stride: {stride}")
    print(f"grid_size: {grid_size}")
    print(f"log_interval: {log_interval}")

    if with_rerun:
        rr.init("Sliding Window BA", recording_id=uuid.uuid4())
        rr.connect()

        rr.log("world", rr.Clear(recursive=True))
        rr.log("log", rr.Clear(recursive=True))
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    seq_len = seq_flow_occs_fwd.shape[0]

    track_len = min(track_len, seq_len)

    initial_depths_fwd, pixel_tracks_fwd, uncerts_fwd, indices_fwd, depths_fwd, rgbs_fwd = compute_pixel_tracks(seq_flow_occs_fwd.cuda(), uncertainties, seq_depths.cuda(), track_len=track_len, stride=stride, grid_size=grid_size, imgs=seq_imgs.cuda(), long_tracks=long_tracks)
    initial_depths_bwd, pixel_tracks_bwd, uncerts_bwd, indices_bwd, depths_bwd, rgbs_bwd = compute_pixel_tracks(seq_flow_occs_bwd.cuda(), uncertainties, seq_depths.cuda(), track_len=track_len, stride=stride, grid_size=grid_size, is_backward=True, imgs=seq_imgs.cuda(), long_tracks=long_tracks)
    initial_depths = torch.cat([initial_depths_fwd, initial_depths_bwd], dim=1)
    pixel_tracks = torch.cat([pixel_tracks_fwd, pixel_tracks_bwd], dim=1)
    uncerts = torch.cat([uncerts_fwd, uncerts_bwd], dim=1)
    indices = torch.cat([indices_fwd, indices_bwd], dim=1)
    depths = torch.cat([depths_fwd, depths_bwd], dim=1) 
    rgbs = torch.cat([rgbs_fwd, rgbs_bwd], dim=1)

    initial_depths = initial_depths_fwd
    pixel_tracks = pixel_tracks_fwd
    uncerts = uncerts_fwd
    indices = indices_fwd
    depths = depths_fwd
    rgbs = rgbs_fwd

    n, wc, gs = initial_depths.shape
    n, wc, gs, tl, c = pixel_tracks.shape
    n, wc, gs, tl, _ = uncerts.shape
    n, wc, gs, tl, _ = indices.shape
    seq_len, c, h, w = seq_imgs.shape

    ba_imgs = seq_imgs
    ba_indices = indices.cuda()
    ba_uncerts = uncerts.cuda()

    ba_proj = np.array(proj.squeeze())
    ba_proj[0, 0] = (ba_proj[0, 0] / w) * 2
    ba_proj[1, 1] = (ba_proj[1, 1] / h) * 2
    ba_proj[0, 2] = (ba_proj[0, 2] / w) * 2 - 1
    ba_proj[1, 2] = (ba_proj[1, 2] / h) * 2 - 1
    ba_proj = torch.tensor(ba_proj).unsqueeze(0).cuda()

    ba_poses_c2w = torch.tensor(np.array(initial_trajectory)).unsqueeze(0).cuda()
    rel_poses = torch.inverse(ba_poses_c2w[:, :-1]) @ ba_poses_c2w[:, 1:]

    ba_param_inv_depth = 1 / initial_depths

    ba_param_rot, ba_param_t = pose_to_param(ba_poses_c2w, rotation_representation)

    ba_param_focal_length = (torch.tensor(proj[0, 0] / w * 2, device=device)).log() / 2

    ba_param_inv_depth.requires_grad = True
    ba_param_rot.requires_grad = True
    ba_param_t.requires_grad = True
    ba_param_focal_length.requires_grad = True

    if with_rerun:
        log_ba_imgs(ba_imgs, uncertainties=uncertainties, tracks=(ba_indices, ba_uncerts, pixel_tracks), timestep=-1)

        log_ba_state(
            param_to_pose(ba_param_rot, ba_param_t),
            imgs=ba_imgs,
            timestep=seq_len + rerun_offset,
            max_dist=10,
        )

    log_step = 1

    optimized_until = 1

    global_ba_step = 1

    last_global_done = False

    print("Starting BA refinement.")
    print("Note: Due to torch.compile, the first iteration might be slow, but the overall speed will be significantly improved after that.")

    while optimized_until < seq_len or not last_global_done:
        do_last_global = optimized_until >= seq_len

        if do_last_global:
            last_global_done = True

        do_global = global_ba_step % global_every_n == 0 or do_last_global

        ba_window_start = max(optimized_until - overlap, 0)
        ba_window_end = min(ba_window_start + ba_window, seq_len)

        seq_ids = torch.arange(seq_len, device=device)

        if not do_global:
            print(f"Optimizing from {ba_window_start} to {ba_window_end}")


            ba_param_inv_depth_mask = (ba_indices[:, :, :, 0, 0] >= optimized_until) & (ba_indices[:, :, :, 0, 0] < ba_window_end)
            ba_param_pose_mask = ((seq_ids >= optimized_until) & (seq_ids < ba_window_end)).view(1, -1, 1)
            loss_mask = (ba_indices[:, :, :, 1:, 0] >= ba_window_start) & (ba_indices[:, :, :, 1:, 0] < ba_window_end)

        else:

            print(f"Optimizing globally util {optimized_until}")

            ba_param_inv_depth_mask = (ba_indices[:, :, :, 0, 0] >= 0) & (ba_indices[:, :, :, 0, 0] < optimized_until)
            ba_param_pose_mask = ((seq_ids >= 0) & (seq_ids < optimized_until)).view(1, -1, 1)
            loss_mask = (ba_indices[:, :, :, 1:, 0] >= 0) & (ba_indices[:, :, :, 1:, 0] < optimized_until)

        if all_reg_to_zero and do_last_global:
            lambda_depth = 0
            lambda_pose = 0


        ba_poses = param_to_pose(ba_param_rot, ba_param_t).detach().clone()

        if optimize_relatives:
            ba_poses_ = [ba_poses[:, 0]]
            for i in range(1, seq_len):
                ba_poses_.append(torch.inverse(ba_poses[:, i-1]) @ ba_poses_[-1])

            ba_poses = torch.stack(ba_poses_, dim=1)

        ba_poses = ba_poses[:, :optimized_until]

        if optimized_until < seq_len:
            add_poses = []

            curr_pose = ba_poses[:, -1]

            for frame_idx in range(optimized_until, seq_len):
                curr_pose = curr_pose @ rel_poses[:, frame_idx - 1]
                add_poses.append(curr_pose)
            
            add_poses = torch.stack(add_poses, dim=1)

            ba_poses = torch.cat([ba_poses, add_poses], dim=1)

        ba_param_rot, ba_param_t = pose_to_param(ba_poses, rotation_representation)

        ba_param_rot = ba_param_rot.detach().clone()
        ba_param_t = ba_param_t.detach().clone()

        ba_param_rot.requires_grad = True
        ba_param_t.requires_grad = True

        optimizer = torch.optim.Adam([ba_param_inv_depth, ba_param_rot, ba_param_t, ba_param_focal_length], lr=lr)

        if not do_global:
            n_steps_ = n_steps_sliding
        else:
            if do_last_global:
                n_steps_ = n_steps_last_global
            else:
                n_steps_ = n_steps_global

        pbar = tqdm(range(n_steps_))

        for step in pbar:
            optimizer.zero_grad()

            # Detach relevant parameters:
            ba_param_inv_depth_d = ba_param_inv_depth.clone()
            ba_param_inv_depth_d[~ba_param_inv_depth_mask].detach_()

            ba_param_rot_d = ba_param_rot.clone()            
            ba_param_rot_d[~ba_param_pose_mask.expand_as(ba_param_rot_d)] = ba_param_rot_d[~ba_param_pose_mask.expand_as(ba_param_rot_d)].detach()
            # ba_param_rot_d[~ba_param_pose_mask.expand_as(ba_param_rot_d)].detach_()

            ba_param_t_d = ba_param_t.clone()
            ba_param_t_d[~ba_param_pose_mask.expand_as(ba_param_t_d)] = ba_param_t_d[~ba_param_pose_mask.expand_as(ba_param_t_d)].detach()
            # ba_param_t_d[~ba_param_pose_mask.expand_as(ba_param_t_d)].detach_()

            repr_loss, smoothness_loss, ba_proj, xyzh_world = compute_loss(ba_param_inv_depth_d, ba_param_rot_d, ba_param_t_d, ba_param_focal_length, pixel_tracks, ba_indices, ba_uncerts, w, h, loss_mask, max_uncert)

            if use_best:
                thresh = torch.quantile(repr_loss[repr_loss > 0], 0.9)
                repr_loss[repr_loss > thresh] = 0

            total_loss = repr_loss.mean() + smoothness_loss.mean() * (lambda_smoothness if do_global else 0)

            total_loss.backward()

            # clip gradients

            for param in [ba_param_inv_depth, ba_param_rot, ba_param_t, ba_param_focal_length]:
                param.grad[torch.isnan(param.grad) | torch.isinf(param.grad)] = 0

            
            if do_global and step < n_steps_only_focal:
                ba_param_rot.grad *= 0
                ba_param_t.grad *= 0

            torch.nn.utils.clip_grad_norm_([ba_param_inv_depth, ba_param_rot, ba_param_t, ba_param_focal_length], .1)


            optimizer.step()

            # pbar.set_postfix({"total_loss": total_loss.item(), "loss": loss.item(), "pose_loss": pose_loss.item(), "depth_loss": depth_loss.item(), "depth_repr_loss": depth_loss_repr.mean().item(),"smoothness_loss": smoothness_loss.item(), "fx": ba_proj[0, 0, 0].item(), "fy": ba_proj[0, 1, 1].item(), "uncert_mean": pt_mean.item(), "uncert_std": pt_std.item()})
            pbar.set_postfix({"l": total_loss.item(), "l_s": smoothness_loss.item(), "fx": ba_proj[0, 0, 0].item(), "fy": ba_proj[0, 1, 1].item()})

            log_step += 1

            if log_step % log_interval == 0 and with_rerun:
                ba_poses_c2w = param_to_pose(ba_param_rot, ba_param_t)

                if optimize_relatives:
                    ba_poses_c2w_ = [ba_poses[:, 0]]
                    for i in range(1, seq_len):
                        ba_poses_c2w_.append(torch.inverse(ba_poses_c2w[:, i-1]) @ ba_poses_c2w_[-1])

                    ba_poses_c2w = torch.stack(ba_poses_c2w_, dim=1)

                log_ba_imgs(ba_imgs, timestep=seq_len + rerun_offset * 2 + (log_step // log_interval), frame_idx=ba_window_end-1)
                
                log_ba_state(
                    ba_poses_c2w,
                    points=xyzh_world[:, ::1, :3, 0].permute(0, 2, 1),
                    point_colors=rgbs[:, :, :, :1, :].reshape(1, -1, 3)[:, ::1],
                    timestep=seq_len + rerun_offset * 2  + (log_step // log_interval),
                    max_dist=10,
                )

        if not do_global:
            optimized_until = ba_window_end

        global_ba_step += 1

    ba_poses_c2w = param_to_pose(ba_param_rot, ba_param_t)

    ba_trajectory = ba_poses_c2w.detach()
    ba_proj = make_normalized_proj((ba_param_focal_length * 2).exp(), w/h)[0].cpu().detach()

    # Unnormalize proj
    ba_proj[0, 0] = (ba_proj[0, 0] / 2) * w
    ba_proj[1, 1] = (ba_proj[1, 1] / 2) * h
    ba_proj[0, 2] = (ba_proj[0, 2] + 1) / 2 * w
    ba_proj[1, 2] = (ba_proj[1, 2] + 1) / 2 * h

    ba_extras = {
        "ba_param_inv_depth": ba_param_inv_depth.detach().cpu(),
        "ba_param_rot": ba_param_rot.detach().cpu(),
        "ba_param_t": ba_param_t.detach().cpu(),
        "ba_param_focal_length": ba_param_focal_length.detach().cpu(),
        "ba_trajectory": ba_trajectory.cpu(),
        "indices": ba_indices.cpu(),
        "uncerts": ba_uncerts.cpu(),
        "initial_depths": initial_depths.cpu(),
        "pixel_tracks": pixel_tracks.cpu(),
    }

    return ba_trajectory[0], ba_proj, ba_extras


@hydra.main(version_base=None, config_path=str("../configs"))
def main(config: DictConfig):

    device = "cuda"
    
    logger.info("Loading model")

    model_path = Path(config.model_path)

    if not model_path.is_dir():
        config_path = model_path.parent
    else:
        config_path = model_path
    name = config_path.name

    training_config = omegaconf.OmegaConf.load(config_path / "training_config.yaml")
    training_config["model"]["use_provided_flow"] = False

    checkpoint_path = get_checkpoint_path(model_path)

    logger.info(f"Loading model from {checkpoint_path}")

    model = load_model(training_config, checkpoint_path).to(device)

    criterion = [make_loss(cfg) for cfg in training_config.get("loss", [])][0]

    logger.info("Loading images")

    input_path = Path(config.input_path)
    imgs = load_images(input_path)

    logger.info("Creating dataset")

    dataset = make_dataset(imgs, device=device)

    if rr is None:
        logger.warning("Rerun is not installed, will not record")
        config.with_rerun = False

    if config.get("with_rerun", False):
        rr.init("Fit Video", recording_id=uuid.uuid4())
        rr.connect()

        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)


    # logger.info("Overfitting model")

    # overfit_model(config, model, criterion, dataset)

    logger.info("Fitting video")

    fit_video(config, model, criterion, dataset)


if __name__ == "__main__":
    main()