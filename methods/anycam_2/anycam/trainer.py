from copy import copy, deepcopy
import logging
from pathlib import Path
import random

import ignite.distributed as idist
from ignite.engine import Engine, Events
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from torch.amp import autocast
from torch.utils.data import WeightedRandomSampler

from dotdict import dotdict

from anycam.common.geometry import get_grid_xy
from anycam.common.scheduler import make_scheduler
from anycam.common.image_processor import  make_image_processor

from anycam.training.base_trainer import base_training

from anycam.datasets import make_datasets
from anycam.models import make_depth_aligner, make_depth_predictor, make_pose_predictor

from anycam.loss import make_loss


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# can_compile = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
can_compile = False

logger = logging.getLogger(__name__)


@torch.compile(disable=not can_compile)
def make_proj_from_focal_length(focal_length, aspect_ratio=1.0):
    proj = torch.eye(3, device=focal_length.device).view(*((1,) * len(focal_length.shape)), 3, 3).repeat(*focal_length.shape, 1, 1)
    proj[..., 0, 0] = focal_length
    proj[..., 1, 1] = focal_length * aspect_ratio

    return proj


@torch.compile(disable=not can_compile, dynamic=True)
def normalize_proj(projs, h, w):
    projs = projs.clone()

    projs[..., 0, 0] = (projs[..., 0, 0] / w) * 2
    projs[..., 1, 1] = (projs[..., 1, 1] / h) * 2

    projs[..., 0, 2] = (projs[..., 0, 2] / w) * 2 - 1
    projs[..., 1, 2] = (projs[..., 1, 2] / h) * 2 - 1

    return projs


@torch.compile(disable=not can_compile, dynamic=True)
def unproject_points(depths, projs):
    # depths: n, 1, h, w
    # projs: n, 3, 3

    n, c, h, w = depths.shape

    device = depths.device

    xyz = get_grid_xy(h, w, device=device, homogeneous=True).expand(n, -1, -1, -1)
    xy = xyz[:, :2]

    inv_projs = torch.inverse(projs.reshape(-1, 3, 3))

    pts = (inv_projs @ xyz.view(n, 3, -1))
    pts = pts * depths.reshape(n, 1, -1)
    pts = torch.cat((pts, torch.ones_like(pts[:, :1])), dim=1)

    return pts, xy


@torch.compile(disable=not can_compile, dynamic=True)
def induce_flow_dist(depths, projs, rel_poses, flow=None, compute_dist=False, return_depth=False):
    # depths: n, f, (1 / num_candidates), 1, h, w
    # projs: n, num_candidates, 3, 3
    # poses: n, f, 4, 4 or n, f, num_candidates, 4, 4
    # Compute the induced flow from the depths, projs and poses
    # Also compute the 3D distance between points in the reference frame and the target frame (the next frame)
    # The last pose should be identity -> Induced flow should be zero
    # Since we need pairs to compute distance, we need to add zeros at the end of the distance tensor

    n, f, _, c, h, w = depths.shape
    _, num_candidates, _, _ = projs.shape
    nfc = n * f * num_candidates    

    depths = depths.view(n, f, -1, c, h, w).expand(-1, -1, num_candidates, -1, -1, -1).reshape(nfc, 1, h, w)
    projs = projs.view(n, 1, num_candidates, 3, 3).expand(-1, f, -1, -1, -1).reshape(nfc, 3, 3)
    rel_poses = rel_poses.view(n, f, -1, 4, 4).expand(-1, -1, num_candidates, -1, -1).contiguous().reshape(nfc, 4, 4)

    with autocast(dtype=torch.float32, device_type="cuda"):

        unproj_pts, xy = unproject_points(depths, projs)

        pts = rel_poses @ unproj_pts

        proj_pts = projs @ pts[:, :3]
        depth = proj_pts[:, 2:3].clamp_min(1e-2)
        proj_pts = proj_pts / depth
        proj_pts = proj_pts.clamp(-2, 2)

        if not return_depth:
            del depth
        else:
            depth = depth.reshape(n, f, num_candidates, 1, h, w)

        induced_flow = proj_pts[:, :2].reshape(-1, 2, h, w) - xy

        induced_flow = induced_flow.view(n, f, num_candidates, 2, h, w)

        if compute_dist:
            unproj_pts = unproj_pts.view(n, f, num_candidates, 4, h, w)[:, :, :, :3]
            pts = pts.view(n, f, num_candidates, 4, h, w)[:, :, :, :3]

            if flow is None:
                # This is a legacy implementation. The code should not come here if you use dist
                corr_xy = (xy.reshape(*induced_flow.shape) + induced_flow).reshape(n, f, num_candidates, 2, h, w)
            else:
                # This should be the correct one
                flow = flow.reshape(n, f, -1, 2, h, w).expand(-1, -1, num_candidates, -1, -1, -1)
                corr_xy = xy.reshape(n, f, num_candidates, 2, h, w) + flow

            corr_pts = F.grid_sample(unproj_pts[:, 1:].reshape(-1, 3, h, w), corr_xy[:, :-1].reshape(-1, 2, h, w).permute(0, 2, 3, 1), align_corners=False)
            corr_pts = corr_pts.reshape(n, f - 1, num_candidates, 3, h, w)

            diffs = pts[:, :-1, :, :3] - corr_pts
            dist = torch.norm(diffs, dim=-3, keepdim=True)
            dist = torch.cat((dist, torch.zeros_like(dist[:, :1])), dim=1)
        else:
            dist = induced_flow[:, :, :, :1].detach().clone()

    if not return_depth:
        return induced_flow, dist
    else:
        return induced_flow, dist, depth


@torch.compile(disable=not can_compile, dynamic=True)
def subsample_pose_input(images, image_features, flow_occs, depths, poses, aligned_depths, drop_n):
    # images: n, f, c, h, w
    # image_features: n, f, c, h, w
    # flow_occs: n, f, 2, h, w
    # depths: n, f, 1, h, w

    n, f, c, h, w = images.shape

    device = images.device

    indices = torch.randperm(f-2, device=device)
    indices = indices[: f - drop_n - 2] + 1
    indices = torch.sort(indices)[0]
    indices = torch.cat((torch.tensor([0], device=device), indices, torch.tensor([f-1], device=device)))
    
    images = images[:, indices]
    image_features = image_features[:, indices]
    depths = depths[:, indices]
    aligned_depths = aligned_depths[:, indices]

    # Chain dropped flows

    flow_occs_parts = []
    flow_occs_sub = []

    xy = get_grid_xy(h, w, device=device, homogeneous=False).expand(n, -1, -1, -1)

    for i in range(f):

        if i in indices and i > 0:
            # Merge accumulated flows and reset

            if len(flow_occs_parts) == 1:
                flow_occs_sub.append(flow_occs_parts[0])
            else:
                acc_flow = flow_occs_parts[0][:, :2]
                acc_occ = flow_occs_parts[0][:, 2:3] > .5

                for j in range(1, len(flow_occs_parts)):
                    flow_occ_resampled = F.grid_sample(flow_occs_parts[j], (xy + acc_flow).permute(0, 2, 3, 1), align_corners=False)
                    acc_flow = acc_flow + flow_occ_resampled[:, :2]
                    acc_occ = acc_occ & (flow_occ_resampled[:, 2:3] > .5)

                acc_flow_occ = torch.cat((acc_flow, acc_occ.to(acc_flow.dtype)), dim=1)
                flow_occs_sub.append(acc_flow_occ)

            flow_occs_parts = []

        flow_occs_parts.append(flow_occs[:, i])

    flow_occs_sub.append(flow_occs_parts[-1])

    flow_occs = torch.stack(flow_occs_sub, dim=1)

    # Chain dropped poses

    poses_parts = []
    poses_sub = []

    for i in range(f):

        if i in indices and i > 0:
            if len(poses_parts) == 1:
                poses_sub.append(poses_parts[0])
            else:
                acc_pose = poses_parts[0]

                for j in range(1, len(poses_parts)):
                    acc_pose = poses_parts[j] @ acc_pose

                poses_sub.append(acc_pose)

            poses_parts = []

        poses_parts.append(poses[:, i])

    poses_sub.append(poses_parts[-1])        

    poses = torch.stack(poses_sub, dim=1)

    return images, image_features, flow_occs, depths, poses, aligned_depths

    
class AnyCamWrapper(nn.Module):
    def __init__(
        self, config
    ) -> None:
        super().__init__()

        self.use_provided_depth = config.get("use_provided_depth", False)
        self.use_provided_flow = config.get("use_provided_flow", False)
        self.use_provided_proj = config.get("use_provided_proj", False)

        self.try_focal_length_candidates = config.get("try_focal_length_candidates", True)

        # Choose between "forward", "backward", "random", "both"
        self.train_directions = config.get("train_directions", "forward")

        self.perform_subsampled_pose_pass = config.get("perform_subsampled_pose_pass", True)
        self.subsampling_drop_n = config.get("subsampling_drop_n", 1)
        self.initialize_subsampled_poses = config.get("initialize_subsampled_poses", False)
        self.initialize_subsampled_focal_length_probs = config.get("initialize_subsampled_focal_length_probs", True)

        self.single_focal_warmup_iters = config.get("single_focal_warmup_iters", 0)

        self.flow_model = config.get("flow_model", "unimatch")

        self.depth_predictor = make_depth_predictor(config["depth_predictor"])
        self.pose_predictor = make_pose_predictor(config["pose_predictor"])
        self.depth_aligner = make_depth_aligner(config["depth_aligner"])

        self.image_processor = make_image_processor({"type": "flow_occlusion"}, flow_model=self.flow_model, use_provided_flow=self.use_provided_flow, pair_mode="sequential")

        self.renderer = dotdict({"net": None})
        self.renderer.net = None

        self.z_near = config.get("z_near", 0.1)
        self.z_far = config.get("z_far", 10)

        self._counter = 0
        
        for param in self.depth_predictor.parameters():
            param.requires_grad = False

    def forward(self, data, tag=None, **kwargs):

        data = dict(data)

        images = data["imgs"]  # B, n_frames, c, h, w
        gt_projs = data["projs"]  # B, n_frames, 3, 3

        n, f, c, h, w = images.shape
        device = images.device

        # Normalize projection matrices
        # We assume that all frames from a sequence have the same projection matrix
        gt_projs = normalize_proj(gt_projs[:, 0], h, w)

        # Get depth and flow either from the dataset (preprocessed) or from the model
        
        if self.use_provided_depth:
            depths = data["depths"]

        else:
            depth_in = images.view(n * f, c, h, w)

            with torch.no_grad():
                depths, depth_features = self.depth_predictor(depth_in, return_features=True)
            depths = depths[0]

            depths = 1 / depths.clamp_min(1e-3).view(n, -1, 1, *depths.shape[-2:])
            depth_features = depth_features.view(n, -1, *depth_features.shape[1:])

        data["pred_depths"] = depths * .1
        data["pred_depths_list"] = [depths]

        images_ip_fwd, images_ip_bwd = self.image_processor(images * 2 - 1, data=data) # Legacy image processor. Requires images in range -1 to 1

        flow_occ_fwd = images_ip_fwd[:, :, 3:6]
        flow_occ_bwd = images_ip_bwd[:, :, 3:6]

        # img_features = self.pose_predictor.get_img_features(images)
        img_features = torch.zeros_like(images)

        # Build input data for pose predictor
        
        if self.train_directions == "forward":
            directions = ["forward"]
        elif self.train_directions == "backward":
            directions = ["backward"]
        elif self.train_directions == "random":
            if torch.rand(1).item() > 0.5:
                directions = ["forward"]
            else:
                directions = ["backward"]
        elif self.train_directions == "both":
            directions = ["forward", "backward"]
        else:
            raise ValueError("Invalid direction selection.")

        images_in = []
        img_features_in = []
        flow_occs_in = []
        depths_in = []

        for direction in directions:
            if direction == "forward":
                images_in.append(images)
                img_features_in.append(img_features)
                flow_occs_in.append(flow_occ_fwd)
                depths_in.append(depths)

            elif direction == "backward":
                images_in.append(torch.flip(images, dims=(1,)))
                img_features_in.append(torch.flip(img_features, dims=(1,)))
                flow_occs_in.append(torch.flip(flow_occ_bwd, dims=(1,)))
                depths_in.append(torch.flip(depths, dims=(1,)))

        images_in = torch.cat(images_in, dim=0)
        img_features_in = torch.cat(img_features_in, dim=0)
        flow_occs_in = torch.cat(flow_occs_in, dim=0)
        depths_in = torch.cat(depths_in, dim=0)

        # Update batch size
        n = images_in.shape[0]

        # Predict poses, uncertainties, focal length, etc.

        pose_result = self.pose_predictor(
            images_in,
            # img_features=img_features_in,
            flow_occs=flow_occs_in,
            depths=depths_in,
            initial_poses=None,
            initial_focal_length_probs=None,
            initial_scaling_feature=None,
        )

        uncert = pose_result["uncert"]
        poses = pose_result["poses"]
        focal_length = pose_result["focal_length"]
        focal_length_candidates = pose_result["focal_length_candidates"]
        focal_length_probs = pose_result["focal_length_probs"]  
        scaling_feature = pose_result["scaling_feature"]

        # This has shape (n, num_candidates, 3, 3)
        # If we only use the provided focal length, we should have only one candidate
        if self.try_focal_length_candidates:
            proj_candidates = make_proj_from_focal_length(focal_length_candidates, w/h)

            if self.single_focal_warmup_iters > 0 and self.training and "iteration" in data:
                curr_iter = data["iteration"][0].item()
                if curr_iter <= self.single_focal_warmup_iters:
                    logger.warning(f"Using single focal length at {curr_iter}/{self.single_focal_warmup_iters} iterations")
                    # num_candidates = focal_length_candidates.shape[1]
                    # proj_candidates = proj_candidates[:, num_candidates//2:num_candidates//2+1].expand(-1, num_candidates, -1, -1)
                    pose_result["poses"][..., :3, :3] = torch.eye(3, device=device).view(1, 1, 1, 3, 3).repeat(*pose_result["poses"].shape[:-2], 1, 1)
                    # uncert = torch.ones_like(uncert)
                    # pose_result["uncert"] = uncert

        else:
            proj_candidates = make_proj_from_focal_length(focal_length.unsqueeze(-1), w/h)

        # If we only use the provided focal length, we replace the candidates with the provided focal length
        if self.use_provided_proj:
            proj_candidates = gt_projs[:, None]

        # If depth aligner is available, we use it to align the depth

        aligned_depths = depths_in.view(-1, f, 1, 1, h, w)
        alignment_params = torch.zeros(aligned_depths.shape[0], f, 1, 1, device=device)

        # Induce flow and compute distance

        induced_flow, dist = induce_flow_dist(aligned_depths, proj_candidates, poses, flow_occs_in[..., :2, :, :])

        pose_result["flow_occs_in"] = flow_occs_in
        pose_result["aligned_depths"] = aligned_depths
        pose_result["alignment_params"] = alignment_params
        pose_result["induced_flow"] = induced_flow
        pose_result["dist"] = dist
        pose_result["proj_candidates"] = proj_candidates

        data["pose_result"] = pose_result

        # Compability with previous implementation
        if self.use_provided_proj:
            best_focal_length_index = torch.zeros(n, device=device, dtype=torch.long)
        else:
            if "target_focal" in kwargs:
                target_focal = kwargs["target_focal"] 
                best_focal_length_index = (pose_result["focal_length_candidates"] - target_focal).abs().argmin(dim=-1)
            else:
                best_focal_length_index = torch.argmax(focal_length_probs[:, 0], dim=-1)

        selected_induced_flow = induced_flow[torch.arange(n, device=device), :, best_focal_length_index, :, :, :]
        selected_proj = proj_candidates[torch.arange(n, device=device), best_focal_length_index][:, None]

        if len(poses.shape) == 4:
            selected_poses = poses
        elif poses.shape[2] == 1:
            selected_poses = poses[:, :, 0]
        else:
            selected_poses = poses[torch.arange(n, device=device), :, best_focal_length_index]
        
        if aligned_depths.shape[2] == 1:
            selected_aligned_depths = aligned_depths[:, :, 0]
        else:
            selected_aligned_depths = aligned_depths[torch.arange(n, device=device), :, best_focal_length_index]
        
        if uncert.shape[2] == 1:
            selected_uncert = uncert[:, :, 0]
        else:
            selected_uncert = uncert[torch.arange(n, device=device), :, best_focal_length_index]

        data["images_ip"] = images_ip_fwd
        data["induced_flow"] = selected_induced_flow
        data["induced_flow_list"] = [selected_induced_flow]
        data["valid"] = images_ip_fwd[:, :, 5:6] > 0.5
        data["proc_poses"] = selected_poses
        data["proc_projs"] = selected_proj
        data["uncertainties"] = selected_uncert
        data["weights_proc"] = selected_uncert
        data["scaled_depths"] = [selected_aligned_depths]

        data["z_near"] = torch.tensor(self.z_near, device=images.device)
        data["z_far"] = torch.tensor(self.z_far, device=images.device)

        if self.training:
            self._counter += 1
            
        return data

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result


def training(local_rank, config):
    return base_training(
        local_rank,
        config,
        get_dataflow,
        initialize,
        get_custom_trainer_events
    )


def get_subset(config, len_dataset: int):
    subset_type = config.get("type", None)
    match subset_type:
        case "random":
            return torch.sort(
                torch.randperm(len_dataset)[: config["args"]["size"]]
            ).tolist()
        case "range":
            return list(
                range(
                    config["args"].get("start", 0),
                    config["args"].get("end", len_dataset),
                )
            )
        case _:
            return list(range(len_dataset))


def get_dataflow(config):
    if idist.get_local_rank() > 0:
        idist.barrier()
    
    train_dataset_list = config["dataset"]
    val_dataset_list = config.get("val_dataset", train_dataset_list)
    vis_dataset_list = config.get("vis_dataset", train_dataset_list)

    dataset_list = list(set(train_dataset_list + val_dataset_list + vis_dataset_list))

    dataset_cfgs = {
        dataset_name: config["dataset_cfgs"][dataset_name] for dataset_name in dataset_list
    }

    dataset_params = config.get("dataset_params", {})
    dataset_params["sequential"] = True

    train_datasets, test_datasets = {}, {}

    for dataset_name, dataset_cfg in dataset_cfgs.items():
        train_dataset, test_dataset = make_datasets(dataset_cfg, **dataset_params)

        train_datasets[dataset_name] = train_dataset
        test_datasets[dataset_name] = test_dataset

    train_dataset = torch.utils.data.ConcatDataset([train_datasets[dataset_name] for dataset_name in train_dataset_list])

    weights = []
    weights_info = []
    
    for dataset_name in train_dataset_list:
        weights.extend([1 / len(train_datasets[dataset_name]) / len(dataset_cfgs)] * len(train_datasets[dataset_name]))
        weights_info.append((dataset_name, len(train_datasets[dataset_name])))

    if config.get("dataloading", {}).get("epoch_length", None):
        epoch_length = config["dataloading"]["epoch_length"]
    else:
        epoch_length = len(weights)

    train_sampler = WeightedRandomSampler(weights, epoch_length, replacement=True)
    train_sampler.weights_info = weights_info

    train_loader = idist.auto_dataloader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        # shuffle=True,
        drop_last=True,
        sampler=train_sampler,
    )

    validation_loaders = {}

    for name, validation_config in config["validation"].items():
        return_flow = True

        if name == "validation":
            selected_datasets = val_dataset_list
            return_flow = False
        else:
            selected_datasets = vis_dataset_list

        for dataset_name in selected_datasets:

            dataset = test_datasets[dataset_name]
            dataset = deepcopy(dataset)
            dataset.return_depth = True
            dataset.full_size_depth = True
            dataset.return_flow = return_flow

            if "subset" in validation_config:
                dataset._datapoints = [dataset._datapoints[i] for i in get_subset(validation_config["subset"], len(dataset))]
                dataset.length = len(dataset._datapoints)
            # else:
                # subset = dataset
            
            validation_loaders[f"{name}/{dataset.NAME}"] = idist.auto_dataloader(
                dataset,
                batch_size=validation_config.get("batch_size", 1),
                num_workers=config["num_workers"],
                shuffle=False,
            )

    if idist.get_local_rank() == 0:
        # Ensure that only local rank 0 download the dataset
        idist.barrier()

    return train_loader, validation_loaders


def get_custom_trainer_events(config):
    trainer_events = []

    if "staged_datasets" in config.get("dataloading"):
        staged_datasets_config = config["dataloading"]["staged_datasets"]

        def update_dataloading_weights(engine: Engine):
            dataloader = engine.state.dataloader

            sampler = dataloader.sampler
            
            if isinstance(sampler, torch.utils.data.distributed.DistributedSampler):
                sampler = sampler.sampler

            curr_epoch = engine.state.epoch
            active_datasets = []

            new_weights = []

            for dataset_name, dataset_len in sampler.weights_info:
                if dataset_name in staged_datasets_config and curr_epoch >= staged_datasets_config[dataset_name]:
                    new_weights += [1 / dataset_len] * dataset_len
                    active_datasets += [dataset_name]
                else:
                    new_weights += [0] * dataset_len

            print("Using datasets: " + ", ".join(active_datasets))

            new_weights = sampler.weights.new_tensor(new_weights)
            new_weights = new_weights / len(active_datasets)

            sampler.weights = new_weights
            
        trainer_events.append((Events.EPOCH_STARTED, update_dataloading_weights))
    
    return trainer_events


def initialize(config: dict):
    # Continue if checkpoint already exists
    if config["training"].get("continue", False):
        prefix = "training_checkpoint_"
        ckpts = Path(config["output"]["path"]).glob(f"{prefix}*.pt")
        # TODO: probably correct logic but please check
        training_steps = [int(ckpt.stem.split(prefix)[1]) for ckpt in ckpts]
        if training_steps:
            config["training"]["resume_from"] = (
                Path(config["output"]["path"]) / f"{prefix}{max(training_steps)}.pt"
            )

    seed = 1
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model_conf = config["model"]

    model = AnyCamWrapper(model_conf)

    model = idist.auto_model(model, find_unused_parameters=True)

    params = [param for name, param in model.named_parameters() if "image_processor" not in name]

    optimizer = optim.Adam(
        params, **config["training"]["optimizer"]["args"]
    )
    optimizer = idist.auto_optim(optimizer)

    lr_scheduler = make_scheduler(config["training"].get("scheduler", {}), optimizer)

    criterion = [make_loss(cfg) for cfg in config.get("loss", [])]

    return model, optimizer, criterion, lr_scheduler
