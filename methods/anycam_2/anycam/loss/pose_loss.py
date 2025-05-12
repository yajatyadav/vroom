import logging
from typing import Callable
import torch
import torch.nn.functional as F
from anycam.common.geometry import resample
from anycam.common.util import normalized_entropy
from anycam.loss.base_loss import BaseLoss

from minipytorch3d.rotation_conversions import (
    matrix_to_quaternion,
    quaternion_to_matrix,
    matrix_to_axis_angle,
    axis_angle_to_matrix,
)

# can_compile = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
can_compile = False

logger = logging.getLogger(__name__)


EPS = 1e-4


def make_flow_error(
    criterion: str,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if criterion == "l2":
        return torch.nn.MSELoss(reduction="none")
    elif criterion == "l1":
        return torch.nn.L1Loss(reduction="none")
    else:
        raise ValueError(f"Unknown flow error: {criterion}")


def make_dist_error(
    criterion: str,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if criterion == "l1":
        return torch.nn.L1Loss(reduction="none")
    elif criterion == "sqrt":
        return lambda a, b: torch.sqrt(torch.nn.L1Loss(reduction="none")(a, b))
    else:
        raise ValueError(f"Unknown dist error: {criterion}")



class PoseLoss(BaseLoss):
    def __init__(self, config) -> None:
        super().__init__(config)

        self.flow_crit = make_flow_error(config.get("flow_criterion", "l1"))
        self.dist_crit = make_dist_error(config.get("dist_criterion", "l1"))

        self.lambda_flow = config.get("lambda_flow", 1)
        self.lambda_dist = config.get("lambda_dist", 1)

        self.use_flow_uncertainty = config.get("use_flow_uncertainty", True)
        self.use_dist_uncertainty = config.get("use_dist_uncertainty", True)

        self.candidate_aggregation = config.get("candidate_aggregation", "mean")
        self.use_flow_labels_after_uncert = config.get("use_flow_labels_after_uncert", True)

        self.lambda_subsampled_pose_consistency = config.get("lambda_subsampled_pose_consistency", 1)
        self.lambda_subsampled_focal_length_probs_consistency = config.get("lambda_subsampled_focal_length_probs_consistency", 0)
        
        self.detach_subsampled_pose_consistency = config.get("detach_subsampled_pose_consistency", True)
        self.detach_subsampled_focal_length_probs_consistency = config.get("detach_subsampled_focal_length_probs_consistency", False)

        self.lambda_depth_regularization = config.get("lambda_depth_regularization", 0)
        self.depth_regularization_target_mean = config.get("depth_regularization_target_mean", 2)

        self.lambda_intercandidate_pose = config.get("lambda_intercandidate_pose", 0)
        self.intercandidate_pose_threshold = config.get("intercandidate_pose_threshold", 1e-8)

        self.lambda_fwd_bwd_consistency = config.get("lambda_fwd_bwd_consistency", 0)

        self.lambda_label_scale = config.get("lambda_label_scale", 10)

        self.pose_token_weight_decay = config.get("pose_token_weight_decay", 0)

        self.log_config(config)
    
    def log_config(self, config):
        logger.info(f"Creating PoseLossV2 with config: {config}")
        logger.info(f"flow_criterion: {config.get('flow_criterion', 'l1')}")
        logger.info(f"dist_criterion: {config.get('dist_criterion', 'l1')}")
        logger.info(f"lambda_flow: {self.lambda_flow}")
        logger.info(f"lambda_dist: {self.lambda_dist}")
        logger.info(f"use_flow_uncertainty: {self.use_flow_uncertainty}")
        logger.info(f"use_dist_uncertainty: {self.use_dist_uncertainty}")
        logger.info(f"lambda_subsampled_pose_consistency: {self.lambda_subsampled_pose_consistency}")
        logger.info(f"lambda_subsampled_focal_length_probs_consistency: {self.lambda_subsampled_focal_length_probs_consistency}")
        logger.info(f"detach_subsampled_pose_consistency: {self.detach_subsampled_pose_consistency}")
        logger.info(f"detach_subsampled_focal_length_probs_consistency: {self.detach_subsampled_focal_length_probs_consistency}")
        logger.info(f"lambda_depth_regularization: {self.lambda_depth_regularization}")
        logger.info(f"depth_regularization_target_mean: {self.depth_regularization_target_mean}")
        logger.info(f"lambda_intercandidate_pose: {self.lambda_intercandidate_pose}")
        logger.info(f"intercandidate_pose_threshold: {self.intercandidate_pose_threshold}")
        logger.info(f"lambda_fwd_bwd_consistency: {self.lambda_fwd_bwd_consistency}")
        logger.info(f"pose_token_weight_decay: {self.pose_token_weight_decay}")
        logger.info(f"lambda_label_scale: {self.lambda_label_scale}")
        logger.info(f"candidate_aggregation: {self.candidate_aggregation}")
        logger.info(f"use_flow_labels_after_uncert: {self.use_flow_labels_after_uncert}")

    def get_loss_metric_names(self) -> list[str]:
        loss_metric_names = [
            "loss",
            "flow_loss",
            "dist_loss",
            "sub_flow_loss",
            "sub_dist_loss",
            "sub_pose_consistency",
            "sub_focal_length_probs_consistency",
            "intercandidate_pose_loss",
            "fwd_bwd_consistency",
            "depth_regularization",
            "pose_token_weight_decay",
            "depth_std",
            "depth_mean",
            "fl_entropy",
            "fl_std",
            "fl_mean",
        ]
        return loss_metric_names

    @torch.compile(disable=not can_compile)
    def compute_pose_loss(self, pose_result):
        flow_occs = pose_result["flow_occs_in"]
        induced_flow = pose_result["induced_flow"]
        dist = pose_result["dist"]
        focal_length_probs = pose_result["focal_length_probs"]
        uncert = pose_result["uncert"]

        # flow_occs: (n, f, 3, h, w)
        # induced_flow: (n, f, nc, 2, h, w)
        # dist: (n, f, nc, 1, h, w)
        # focal_length_probs: (n, nc)
        # uncert: (n, f, (1 / num_candidates), 2, h, w)

        flow_occs = flow_occs[:, :-1]
        induced_flow = induced_flow[:, :-1]
        dist = dist[:, :-1]
        uncert = uncert[:, :-1]

        n, f, nc, _, h, w = induced_flow.shape
        device = induced_flow.device

        losses = {}
        extra_return_data = {}

        if nc > 1:
            focal_length_probs = focal_length_probs.view(n, 1, nc, 1, 1, 1)
        else:
            focal_length_probs = torch.ones(n, 1, nc, 1, 1, 1, device=induced_flow.device)

        invalid = (flow_occs[:, :, 2:3] < .5)

        if self.lambda_flow > 0:
            tgt_flow = flow_occs[:, :, :2].view(n, f, 1, 2, h, w).expand(-1, -1, nc, -1, -1, -1)

            induced_flow = induced_flow.clamp(-1, 1)

            flow_loss = (
                self.flow_crit(
                    induced_flow.reshape(-1, 2, h, w),
                    tgt_flow.reshape(-1, 2, h, w),
                )
                .mean(dim=1, keepdim=True)
            )

            flow_loss = flow_loss.reshape(n, f, nc, 1, h, w).to(torch.float32)

            # Hack: Ignore last two candidates
            # flow_loss[:, :, -2:] = EPS

            losses["flow_loss"] = flow_loss.mean().detach()

            flow_loss_pre_uncert = flow_loss.clone().detach()

            if self.use_flow_uncertainty:
                flow_uncert = uncert[..., :1, :, :].view(n, f, -1, 1, h, w).expand(-1, -1, nc, -1, -1, -1).to(torch.float32)
                flow_uncert = flow_uncert.clamp_min(EPS) # (Not necessary)

                flow_loss = flow_loss * (2 ** .5) / (flow_uncert + EPS) + (flow_uncert + EPS).log()

            flow_loss[invalid.unsqueeze(2).expand(n, f, nc, 1, h, w)] = 0
            flow_loss_pre_uncert[invalid.unsqueeze(2).expand(n, f, nc, 1, h, w)] = 0

            flow_loss[torch.isinf(flow_loss) | torch.isnan(flow_loss)] = 0

            extra_return_data["sequence_flow_loss"] = flow_loss.detach().mean(dim=(1, 3, 4, 5))

            if self.candidate_aggregation == "mean":
                # label = torch.argmin(flow_loss.mean(dim=(1, 3, 4, 5)), dim=-1).detach()
                if self.use_flow_labels_after_uncert:
                    soft_label = F.softmax(-self.lambda_label_scale * flow_loss.detach().mean(dim=(1, 3, 4, 5)), dim=-1)
                else:
                    soft_label = F.softmax(-self.lambda_label_scale * flow_loss_pre_uncert.detach().mean(dim=(1, 3, 4, 5)), dim=-1)

                extra_return_data["flow_soft_label"] = soft_label

                flow_loss = flow_loss.mean(dim=2)

                # flow_cls_loss = F.cross_entropy(focal_length_probs.view(n, nc), label)
                flow_cls_loss = F.kl_div(focal_length_probs.view(n, nc).log(), soft_label, reduction="batchmean")
                flow_cls_loss[torch.isinf(flow_cls_loss) | torch.isnan(flow_cls_loss)] = 0
            elif self.candidate_aggregation == "predicted":
                flow_loss = (flow_loss * focal_length_probs).sum(dim=2)

                flow_cls_loss = torch.tensor(0.0, device=device)

            flow_loss = flow_loss.mean() + flow_cls_loss.mean()

        else:
            flow_loss = torch.tensor(0.0, device=device)

            losses["flow_loss"] = 0

        if self.lambda_dist > 0:

            dist_loss = (
                self.dist_crit(
                    dist.reshape(-1, 1, h, w),
                    torch.zeros_like(dist).reshape(-1, 1, h, w),
                )
                .mean(dim=1, keepdim=True)
            )

            dist_loss = dist_loss.reshape(n, f, nc, 1, h, w)

            # dist_loss = dist_loss.detach()
            # logger.error("Remove this line!!")

            losses["dist_loss"] = dist_loss.mean().detach()

            if self.use_dist_uncertainty:
                dist_uncert = uncert[..., 1:, :, :].view(n, f, -1, 1, h, w).expand(-1, -1, nc, -1, -1, -1) * 0 + 1

                dist_loss = dist_loss * (2 ** .5) / (dist_uncert + EPS) + (dist_uncert + EPS).log()

            dist_loss[invalid.unsqueeze(2).expand(n, f, nc, 1, h, w)] = 0

            extra_return_data["sequence_dist_loss"] = dist_loss.detach().mean(dim=(1, 3, 4, 5))

            if self.candidate_aggregation == "mean":
                # label = torch.argmin(dist_loss.mean(dim=(1, 3, 4, 5)), dim=-1).detach()
                soft_label = F.softmax(-10 * dist_loss.detach().mean(dim=(1, 3, 4, 5)), dim=-1)
                dist_loss = dist_loss.mean(dim=2)

                extra_return_data["dist_soft_label"] = soft_label

                # dist_cls_loss = F.cross_entropy(focal_length_probs.view(n, nc), label)
                dist_cls_loss = F.kl_div(focal_length_probs.view(n, nc).log(), soft_label, reduction="batchmean")
            elif self.candidate_aggregation == "predicted":
                dist_loss = (dist_loss * focal_length_probs).sum(dim=2)

                dist_cls_loss = torch.tensor(0.0, device=device)

            dist_loss = dist_loss.mean() + dist_cls_loss.mean()

        else:
            dist_loss = torch.tensor(0.0, device=device)

            losses["dist_loss"] = 0

        loss = self.lambda_flow * flow_loss + self.lambda_dist * dist_loss

        return loss, losses, extra_return_data

    @torch.compile(disable=not can_compile)
    def compute_consistency_loss(self, sub_pose_result):

        if self.lambda_subsampled_pose_consistency > 0:
            tgt_poses = sub_pose_result["poses_in"]
            poses = sub_pose_result["poses"]

            n, f, num_candidates, _, _ = poses.shape

            if self.detach_subsampled_pose_consistency:
                tgt_poses = tgt_poses.detach()

            diff = torch.inverse(poses) @ tgt_poses
            id = torch.eye(4, device=diff.device).view(1, 1, 1, 4, 4).expand(n, f, num_candidates, -1, -1)

            pose_consistency_loss = (diff - id).abs().mean(dim=(2, 3)).mean()

        else:
            pose_consistency_loss = torch.tensor(0.0, device=poses.device)
        
        if self.lambda_subsampled_focal_length_probs_consistency > 0:
            tgt_focal_length_probs = sub_pose_result["focal_length_probs_in"]
            focal_length_probs = sub_pose_result["focal_length_probs"]

            focal_length_probs_consistency_loss = F.kl_div(
                focal_length_probs.log(),
                tgt_focal_length_probs,
                reduction="batchmean",
            ).mean()

        else:
            focal_length_probs_consistency_loss = torch.tensor(0.0, device=pose_consistency_loss.device)

        loss = (
            self.lambda_subsampled_pose_consistency * pose_consistency_loss
            + self.lambda_subsampled_focal_length_probs_consistency * focal_length_probs_consistency_loss
        )

        losses = {
            "pose_consistency": pose_consistency_loss.detach(),
            "focal_length_probs_consistency": focal_length_probs_consistency_loss.detach(),
        }

        return loss, losses
    
    @torch.compile(disable=not can_compile)
    def compute_regularization_loss(self, pose_result):
        aligned_depths = pose_result["aligned_depths"]

        if self.lambda_depth_regularization > 0:
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                inv_aligned_depths = 1 / aligned_depths.clamp_min(1e-4)

                inv_aligned_depths_mean = inv_aligned_depths.mean(dim=(-2, -1))

                depth_regularization_loss = torch.abs(inv_aligned_depths_mean - self.depth_regularization_target_mean).mean()

        else:
            depth_regularization_loss = torch.tensor(0.0, device=aligned_depths.device)

        losses = {
            "depth_regularization": depth_regularization_loss.detach(),
        }

        loss = self.lambda_depth_regularization * depth_regularization_loss

        return loss, losses
    
    @torch.compile(disable=not can_compile)
    def compute_intercandidate_pose_loss(self, pose_result, extra_return_data):
        if self.lambda_intercandidate_pose > 0:
            poses = pose_result["poses"]
            num_candidates = pose_result["focal_length_probs"].shape[-1]
            soft_label = extra_return_data["flow_soft_label"]

            n, f, _, _, _ = poses.shape
            id = torch.eye(4, device=poses.device, dtype=poses.dtype).view(1, 1, 1, 4, 4).expand(n, f, num_candidates-1, -1, -1)


            mask = soft_label < self.intercandidate_pose_threshold
            mask = mask.detach()
            mask = mask.view(n, 1, num_candidates).expand(-1, f, -1)

            # Shift left
            shift_poses_tgt = poses[:, :, :-1].detach()
            shift_poses = poses[:, :, 1:]
            mask_tgt = mask[:, :, :-1]
            shift_mask = mask[:, :, 1:] & (~mask_tgt)

            diff = torch.inverse(shift_poses_tgt) @ shift_poses
            left_pose_loss = (diff - id).abs().mean(dim=(-2, -1))
            left_pose_loss[~shift_mask] = 0

            # Shift right
            shift_poses_tgt = poses[:, :, 1:].detach()
            shift_poses = poses[:, :, :-1]
            mask_tgt = mask[:, :, 1:]
            shift_mask = mask[:, :, :-1] & (~mask_tgt)

            diff = torch.inverse(shift_poses_tgt) @ shift_poses
            right_pose_loss = (diff - id).abs().mean(dim=(-2, -1))
            right_pose_loss[~shift_mask] = 0

            pose_loss = left_pose_loss.mean() + right_pose_loss.mean()

            # mean_pose = (poses * ((~mask).to(poses.dtype).view(n, f, num_candidates, 1, 1))).sum(dim=2, keepdim=True) / (~mask).to(poses.dtype).sum(dim=2, keepdim=True).view(n, f, 1, 1, 1)
            # mean_pose = mean_pose.detach()

            # pose_loss = (poses - mean_pose).abs().mean(dim=(-2, -1))
            # pose_loss[~mask] = 0

            # pose_loss = pose_loss.mean()

        else:
            pose_loss = torch.tensor(0.0, device=pose_result["poses"].device)

        losses = {
            "intercandidate_pose": pose_loss.detach(),
        }

        pose_loss = self.lambda_intercandidate_pose * pose_loss

        return pose_loss, losses

    def compute_forward_backward_consistency(self, pose_result):
        poses = pose_result["poses"]

        if self.lambda_fwd_bwd_consistency > 0:
            n, f, nc, _, _ = poses.shape

            # In forward backward training, pairs are stacked along the batch dimension

            poses_fwd = poses[:n//2]
            poses_bwd = poses[n//2:]

            poses_fwd = poses_fwd[:, :-1]
            poses_bwd = poses_bwd[:, :-1]

            # Backward sequences need to be reversed

            poses_bwd = torch.flip(poses_bwd, dims=(1,))

            diff = poses_fwd @ poses_bwd

            diff_rot = matrix_to_quaternion(diff[..., :3, :3])
            diff_trans = diff[..., :3, 3]

            identity_rot = torch.tensor([1, 0, 0, 0], device=diff_rot.device, dtype=diff_rot.dtype).view(1, 1, 1, 4)
            
            loss_rot = (diff_rot - identity_rot).norm(dim=-1)
            loss_trans = diff_trans.norm(dim=-1)

            loss = loss_rot.mean() + loss_trans.mean()

            # id = torch.eye(4, device=diff.device).view(1, 1, 1, 4, 4).expand(n//2, f-1, nc, -1, -1)

            # loss = (diff - id).abs().mean(dim=(-2, -1)).mean()
        else:
            loss = torch.tensor(0.0, device=poses.device)
        
        losses = {
            "fwd_bwd_consistency": loss.detach(),
        }

        loss = self.lambda_fwd_bwd_consistency * loss

        return loss, losses

    def compute_pose_token_weight_decay(self, pose_result):
        if self.pose_token_weight_decay > 0:
            pose_token_1 = pose_result["wd_pose_token_1"]
            pose_token_2 = pose_result["wd_pose_token_2"]


            wd_loss_1 = (pose_token_1.abs() - 1).clamp_min(0) ** 2
            wd_loss_1 = wd_loss_1.mean()

            wd_loss_2 = (pose_token_2.abs() - 1).clamp_min(0) ** 2
            wd_loss_2 = wd_loss_2.mean()

            loss = self.pose_token_weight_decay * (wd_loss_1 + wd_loss_2)

        else:
            loss = torch.tensor(0.0, device=pose_result["poses"].device)

        losses = {
            "pose_token_weight_decay": loss.detach(),
        }

        return loss, losses

    @torch.compile(disable=not can_compile)
    def compute_metrics(self, pose_result):

        with torch.autocast(device_type="cuda", dtype=torch.float32):
            depth = pose_result["aligned_depths"]
            inv_depth = 1 / depth.clamp_min(1e-4)
            depth_std, depth_mean = torch.std_mean(inv_depth)

        fl_entropy = torch.mean(normalized_entropy(pose_result["focal_length_probs"], dim=-1))

        fl_std, fl_mean = torch.std_mean(pose_result["focal_length"])

        if torch.isnan(fl_std):
            fl_std = torch.tensor(0.0, device=fl_std.device)
        
        if torch.isnan(fl_mean):
            fl_mean = torch.tensor(0.0, device=fl_mean.device)

        metrics = {
            "depth_std": depth_std.detach(),
            "depth_mean": depth_mean.detach(),
            "fl_entropy": fl_entropy.detach(),
            "fl_std": fl_std.detach(),
            "fl_mean": fl_mean.detach(),
        }

        return metrics

    def __call__(self, data, **kwargs):
        pose_result = data["pose_result"]

        # Compute main loss
        loss, losses, extra_return_data = self.compute_pose_loss(pose_result)

        # Compute subsampled loss
        if "sub_pose_result" in data:
            sub_pose_result = data["sub_pose_result"]

            sub_loss, sub_losses, sub_extra_return_data = self.compute_pose_loss(sub_pose_result)

            loss = loss + sub_loss

            for k, v in sub_losses.items():
                losses["sub_" + k] = v

            sub_consistency_loss, sub_consistency_losses = self.compute_consistency_loss(sub_pose_result)

            loss = loss + sub_consistency_loss

            for k, v in sub_consistency_losses.items():
                losses["sub_" + k] = v

        else:
            losses["sub_flow_loss"] = 0
            losses["sub_dist_loss"] = 0
            losses["sub_pose_consistency"] = 0
            losses["sub_focal_length_probs_consistency"] = 0

        # Compute intercandidate pose loss
        intercandidate_pose_loss, intercandidate_pose_losses = self.compute_intercandidate_pose_loss(pose_result, extra_return_data)

        loss = loss + intercandidate_pose_loss

        for k, v in intercandidate_pose_losses.items():
            losses[k] = v

        # Compute forward backward consistency loss
        fwd_bwd_consistency_loss, fwd_bwd_consistency_losses = self.compute_forward_backward_consistency(pose_result)

        loss = loss + fwd_bwd_consistency_loss

        for k, v in fwd_bwd_consistency_losses.items():
            losses[k] = v

        # Compute regularization loss

        regularization_loss, regularization_losses = self.compute_regularization_loss(pose_result)

        loss = loss + regularization_loss

        for k, v in regularization_losses.items():
            losses[k] = v

        # Compute pose token weight decay loss

        wd_loss, wd_losses = self.compute_pose_token_weight_decay(pose_result)
        
        loss = loss + wd_loss

        for k, v in wd_losses.items():
            losses[k] = v

        # Compute metrics
        metrics = self.compute_metrics(pose_result)

        for k, v in metrics.items():
            losses[k] = v

        losses["loss"] = loss

        if kwargs.get("return_extra_data", False):
            return losses, extra_return_data
        else:
            return losses
