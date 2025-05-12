import logging
from collections import defaultdict
from dataclasses import field, dataclass

import math
import os
import numpy as np

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast

from transformers.models.depth_anything.modeling_depth_anything import DepthAnythingForDepthEstimation, DepthAnythingConfig
from transformers.models.dinov2.modeling_dinov2 import Dinov2Backbone


from minipytorch3d.rotation_conversions import (
    matrix_to_quaternion,
    quaternion_to_matrix,
    matrix_to_axis_angle,
    axis_angle_to_matrix,
)

from anycam.models.anycam_blocks import (
    Depth_Anything_V2_Small_hf, 
    AnyCamPoseTokenReassembleStage, 
    AnyCamPoseTokenFusionStage, 
    AnyCamPoseTokenHead,
    pose_scaling_linear,
    pose_scaling_quadratic,
)

from anycam.models.anycam_blocks import AttnBlock, CrossAttnBlock, PoseEmbedding

logger = logging.getLogger(__name__)


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

LOG_FOCAL_LENGTH_BIAS = 1.8


class AnyCam(DepthAnythingForDepthEstimation):
    def __init__(
        self,
        config,
    ):
        self.rotation_parameterization = config.get("rotation_parameterization", "quaternion")
        self.focal_parameterization = config.get("focal_parameterization", "candidates")
        self.focal_min = config.get("focal_min", 0.1)
        self.focal_max = config.get("focal_max", 4.0)
        self.focal_num_candidates = config.get("focal_num_candidates", 32)

        self.separate_pose_candidates = config.get("separate_pose_candidates", False)
        self.separate_scaling_candidates = config.get("separate_scaling_candidates", False)
        self.separate_uncertainty_candidates = config.get("separate_uncertainty_candidates", False)

        self.two_tokens_per_pose = config.get("two_tokens_per_pose", False)
        
        self.scaling_feature_dim = config.get("scaling_feature_dim", 16)

        self.out_uncertainty_dim = config.get("out_uncertainty_dim", 2)

        self.self_att_depth = config.get("self_att_depth", 8)


        self.downsize_input = config.get("downsize_input", None)

        self.use_flow_input = config.get("use_flow_input", True)
        self.use_depth_input = config.get("use_depth_input", True)

        self.pose_token_partial_dropout = config.get("pose_token_partial_dropout", 0.0)
        self.pose_token_dropout = config.get("pose_token_dropout", 0.0)

        if self.rotation_parameterization == "quaternion":
            self.pose_enc_dim = 7
        elif self.rotation_parameterization == "axis-angle":
            self.pose_enc_dim = 6
        else:
            raise ValueError(f"Unknown rotation parameterization: {self.rotation_parameterization}")
        
        if self.focal_parameterization == "log-candidates" or self.focal_parameterization == "linlog-candidates" or self.focal_parameterization == "candidates":
            self.focal_enc_dim = self.focal_num_candidates
        elif self.focal_parameterization == "log":
            self.focal_enc_dim = 1
        else:
            raise ValueError(f"Unknown focal parameterization: {self.focal_parameterization}")

        self.backbone_type = config.get("backbone_type", "dinov2")

        # The default DepthAnything configuration will yield the small DepthAnything model
        self.da_config = DepthAnythingConfig(**Depth_Anything_V2_Small_hf)
        self.da_config.fusion_hidden_size = 128

        if self.backbone_type == "dinov2":
            pass
        elif self.backbone_type == "croco":
            self.da_config.reassemble_hidden_size = 768
            self.da_config.patch_size = 16
            self.downsize_input = (224, 224)

        super().__init__(self.da_config)

        if self.backbone_type == "dinov2":
            # Makes sure that backbone is pretrained
            self.backbone = Dinov2Backbone.from_pretrained("facebook/dinov2-small", **Depth_Anything_V2_Small_hf["backbone_config"])
        elif self.backbone_type == "croco":
            from anycam.models.croco_wrapper import CroCoExtractor

            ckpt_path = os.path.join(os.environ["HOME"], ".cache", "torch", "checkpoints", "CroCo_V2_ViTBase_BaseDecoder.pth")
            logger.info(f"Loading pretrained model from {ckpt_path}")
            if not os.path.exists(ckpt_path):
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                logger.info(f"Downloading from https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo_V2_ViTBase_BaseDecoder.pth")
                r = requests.get('https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo_V2_ViTBase_BaseDecoder.pth')
                with open(ckpt_path , 'wb') as f:
                    f.write(r.content)

            checkpoint = torch.load(ckpt_path)

            self.backbone = CroCoExtractor(**checkpoint["croco_kwargs"])
            self.backbone.load_state_dict(checkpoint["model"], strict=False)

        # Adjust head to predict uncertainty rather than depth
        self.head.max_depth = 1.0
        self.head.activation2 = nn.Identity()
        self.head.conv3 = nn.Conv2d(
            self.da_config.head_hidden_size, 
            self.out_uncertainty_dim * (1 if not self.separate_uncertainty_candidates else self.focal_num_candidates), 
            kernel_size=1, 
            stride=1, 
            padding=0
        )

        # Add pose branch

        self.pose_reassemble_stage = AnyCamPoseTokenReassembleStage(
            self.da_config.reassemble_hidden_size,
            self.da_config.fusion_hidden_size,
            self.da_config.neck_hidden_sizes,
        )
        self.pose_feature_fusion_stage = AnyCamPoseTokenFusionStage(
            self.da_config.fusion_hidden_size,
        )

        self.pose_interframe_attention = nn.ModuleList(
            [
                AttnBlock(
                    self.da_config.fusion_hidden_size,
                    num_heads=4,
                    mlp_ratio=4,
                    attn_class=nn.MultiheadAttention,
                )
                for _ in range(self.self_att_depth)
            ]
        )

        self.sequence_token_attention = CrossAttnBlock(
            self.da_config.fusion_hidden_size,
            self.da_config.fusion_hidden_size,
            num_heads=4,
            mlp_ratio=4
        )

        self.pose_head = AnyCamPoseTokenHead(
            self.da_config.fusion_hidden_size * (1 if not self.two_tokens_per_pose else 2),
            self.pose_enc_dim * (1 if not self.separate_pose_candidates else self.focal_num_candidates),
        )

        self.sequence_info_head = AnyCamPoseTokenHead(
            self.da_config.fusion_hidden_size,
            self.focal_enc_dim + self.scaling_feature_dim * (1 if not self.separate_scaling_candidates else self.focal_num_candidates),
        )

        self.sequence_token = nn.Parameter(torch.randn(1, 1, self.da_config.fusion_hidden_size))

        self.seq_embedding = PoseEmbedding(
            target_dim=1,
            n_harmonic_functions=4,
            append_input=False,
        )

        self.pose_factor = config.get("pose_factor", 0.01)
        pose_scale_function_name = config.get("pose_scaling_function", "linear")
        self.pose_scale_function = globals()[f"pose_scaling_{pose_scale_function_name}"]()

        # Adjust dino projection layer to have more input channels

        if not self.backbone_type == "croco":
            if self.use_flow_input or self.use_depth_input:
                d_in = 6
            else:
                d_in = 3
            
            self.backbone.embeddings.patch_embeddings.projection.weight = nn.Parameter(self.backbone.embeddings.patch_embeddings.projection.weight.repeat(1, d_in // 3, 1, 1))
            self.backbone.embeddings.patch_embeddings.num_channels = d_in

        for name, value in (
            ("_resnet_mean", _RESNET_MEAN),
            ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name,
                torch.FloatTensor(value).view(1, 3, 1, 1),
                persistent=False,
            )

    def prepare_inputs_for_forward(self, images_ip):
        n, f, c, h, w = images_ip.shape

        images_ip = images_ip.reshape(n * f, c, h, w)

        rgb = images_ip[:, :3]
        rest = images_ip[:, 3:]

        base_h = h
        base_w = w

        if self.downsize_input is not None:
            if type(self.downsize_input) == int:
                if base_h < base_w:
                    base_h = self.downsize_input
                    base_w = int(base_h * w / h)
                else:
                    base_w = self.downsize_input
                    base_h = int(base_w * h / w)
            else:
                base_h, base_w = self.downsize_input

        th = math.ceil(base_h / self.da_config.patch_size) * self.da_config.patch_size
        tw = math.ceil(base_w / self.da_config.patch_size) * self.da_config.patch_size

        if h != th or w != tw:
            rgb = F.interpolate(
                rgb, (th, tw), mode="bilinear", align_corners=True
            )
            rest = F.interpolate(
                rest, (th, tw), mode="nearest"
            )

        rgb = (rgb - self._resnet_mean) / self._resnet_std

        images_ip = torch.cat([rgb, rest], dim=1)
        images_ip = images_ip.reshape(n, f, c, th, tw)

        return images_ip
    
    def forward(
        self,
        images,
        flow_occs=None,
        depths=None,
        img_features=None,
        initial_poses=None,
        initial_focal_length_probs=None,
        initial_scaling_feature=None,
    ):
        """
        reshaped_image: Bx3xHxW. The values of reshaped_image are within [0, 1]
        preliminary_cameras: cameras in opencv coordinate.
        """

        inputs = [images]

        if self.use_flow_input and not self.use_depth_input:
            inputs += [flow_occs]
        elif self.use_flow_input and self.use_depth_input:
            inputs += [flow_occs[:, :, :2], depths] # type: ignore
        elif not self.use_flow_input and self.use_depth_input:
            inputs += [depths.expand(-1, -1, 3, -1, -1)] # type: ignore
        
        inputs = torch.cat(inputs, dim=2)

        n, f, c, h, w = inputs.shape

        
        inputs = self.prepare_inputs_for_forward(inputs)

        th, tw = inputs.shape[-2:]

        # Get the 2D image features
        if img_features is None:
            if self.backbone_type == "dinov2":
                outputs = self.backbone.forward_with_filtered_kwargs(inputs.reshape(n * f, c, th, tw), output_hidden_states=False, output_attentions=False)
                hidden_states = outputs.feature_maps
            elif self.backbone_type == "croco":
                img1 = inputs[:, :-1, :3]
                img2 = inputs[:, 1:, :3]

                # Temporary fix to get f feature maps, even though we only need the first f-1

                img1 = img1.reshape(n * (f-1), 3, th, tw)
                img2 = img2.reshape(n * (f-1), 3, th, tw)

                hidden_states = self.backbone(img1, img2)

                hidden_states = [hs.unflatten(0, (n, f-1)) for hs in hidden_states]
                hidden_states = [torch.cat([hs, torch.zeros_like(hs[:, :1])], dim=1) for hs in hidden_states]
                hidden_states = [hs.flatten(0, 1) for hs in hidden_states]

        else:
            hidden_states = img_features

        pose_tokens = [hs[:, 0] for hs in hidden_states]

        patch_size = self.config.patch_size
        patch_height = th // patch_size
        patch_width = tw // patch_size

        # Predict uncertainties
        hidden_states = self.neck(hidden_states, patch_height, patch_width)
        uncertainty = self.head(hidden_states, patch_height, patch_width)

        if th != h or tw != w or self.downsize_input is not None:
            uncertainty = F.interpolate(
                uncertainty, (h, w), mode="bilinear", align_corners=True
            )

        uncertainty = F.softplus(uncertainty)
        uncertainty = uncertainty.reshape(n, f, -1, self.out_uncertainty_dim, h, w)

        # Predict poses
        pose_tokens = self.pose_reassemble_stage(pose_tokens)
        pose_tokens = self.pose_feature_fusion_stage(pose_tokens)
        pose_token = pose_tokens[-1]

        wd_pose_token_1 = pose_token.clone()

        # Perform self-attention

        pose_token = pose_token.reshape(n, f, pose_token.shape[-1])

        # Perform partial dropout
        if self.pose_token_partial_dropout > 0:
            if self.training:
                pose_token = F.dropout(pose_token, p=self.pose_token_partial_dropout, training=self.training)
            else:
                pose_token = pose_token * (1 - self.pose_token_partial_dropout)

        # Add sequence index

        idx = torch.linspace(0, 1, f, device=pose_token.device).view(1, f, 1).expand(n, -1, -1)
        if self.training:
            idx = idx + torch.randn_like(idx) * 0.05
        seq_embedding = torch.zeros_like(pose_token)
        seq_embedding[:, :, :self.seq_embedding.out_dim] = self.seq_embedding(idx).view(n, f, -1)
        
        pose_token = pose_token + seq_embedding

        for i in range(self.self_att_depth):
            pose_token = self.pose_interframe_attention[i](pose_token)

        # Add sequence token

        seq_token = self.sequence_token.expand(n, 1, -1)

        seq_token = self.sequence_token_attention(seq_token, pose_token)

        if self.two_tokens_per_pose:
            pose_token = torch.cat((pose_token, pose_token.roll(-1, dims=1)), dim=-1)

        pose_token = pose_token.reshape(n * f, -1, pose_token.shape[-1])

        wd_pose_token_2 = pose_token.clone()

        with autocast(enabled=True, dtype=torch.float32, device_type="cuda"):
            pose_enc = self.pose_head(pose_token.to(torch.float32))

            pose_enc = pose_enc.view(n, f, -1, self.pose_enc_dim)

            pose_enc_scaled = self.pose_scale_function(pose_enc)

            pose = self.encoding_to_pose(pose_enc_scaled)

            seq_enc = self.sequence_info_head(seq_token.to(torch.float32))
            focal_enc = seq_enc[..., :self.focal_enc_dim]
            scaling_feature = seq_enc[..., self.focal_enc_dim:]

            scaling_feature = scaling_feature.view(n, -1, self.scaling_feature_dim)

            focal_length, focal_length_probs, focal_candidates = self.enc_embed_to_focal(focal_enc)

        pose_result = {
            "uncert": uncertainty,
            "poses": pose,
            "focal_length": focal_length,
            "focal_length_probs": focal_length_probs,
            "focal_length_candidates": focal_candidates,
            "scaling_feature": scaling_feature,
            "wd_pose_token_1": wd_pose_token_1,
            "wd_pose_token_2": wd_pose_token_2,
        }

        return pose_result

    def get_img_features(self, images, depths=None, flow_occs=None):
        inputs = [images]

        if self.use_flow_input and not self.use_depth_input:
            inputs += [flow_occs]
        elif self.use_flow_input and self.use_depth_input:
            inputs += [flow_occs[:, :, :2], depths] # type: ignore
        elif not self.use_flow_input and self.use_depth_input:
            inputs += [depths.expand(-1, -1, 3, -1, -1)] # type: ignore
        
        inputs = torch.cat(inputs, dim=2)

        n, f, c, h, w = inputs.shape
        
        inputs = self.prepare_inputs_for_forward(inputs)

        th, tw = inputs.shape[-2:]

        # Get the 2D image features
        outputs = self.backbone.forward_with_filtered_kwargs(inputs.reshape(n * f, c, th, tw), output_hidden_states=False, output_attentions=False)
        hidden_states = outputs.feature_maps

        return hidden_states
    
    def encoding_to_pose(self, pose_enc):
        n, f, nc, _ = pose_enc.shape

        translation = pose_enc[..., :3]
        rotation = pose_enc[..., 3:]

        # print(f"T={translation[0, 0, 16].cpu().detach().numpy()}, R={rotation[0, 0, 16].cpu().detach().numpy()}")

        if self.rotation_parameterization == "quaternion":
            rotation = quaternion_to_matrix(rotation)
        elif self.rotation_parameterization == "axis-angle":
            rotation = axis_angle_to_matrix(rotation)

        pose = torch.eye(4, device=pose_enc.device).view(1, 1, 1, 4, 4).repeat(n, f, nc, 1, 1)

        pose[..., :3, :3] = rotation
        pose[..., :3, 3] = translation

        return pose
    
    def enc_embed_to_focal(self, focal_enc):
        if self.focal_parameterization == "log":
            focal_length = (focal_enc + LOG_FOCAL_LENGTH_BIAS).exp().view(-1).clamp(self.focal_min, self.focal_max)
            focal_length_probs = None
            focal_candidates = None

        elif self.focal_parameterization == "log-candidates":
            focal_enc = focal_enc * 0.01

            focal_length_probs = F.softmax(focal_enc, dim=-1)
            focal_candidates = torch.linspace(
                math.log(self.focal_min) - LOG_FOCAL_LENGTH_BIAS, 
                math.log(self.focal_max) - LOG_FOCAL_LENGTH_BIAS, 
                self.focal_num_candidates, 
                device=focal_length_probs.device)
            focal_candidates = (focal_candidates + LOG_FOCAL_LENGTH_BIAS).exp()
            focal_candidates = focal_candidates.view(1, -1).expand(focal_length_probs.shape[0], -1)

            focal_length = torch.sum(focal_length_probs * focal_candidates, dim=-1)

        elif self.focal_parameterization == "linlog-candidates":
            focal_enc = focal_enc * 0.01

            focal_length_probs = F.softmax(focal_enc, dim=-1)
            focal_candidates_log = torch.linspace(math.log(self.focal_min), math.log(self.focal_max), self.focal_num_candidates, device=focal_length_probs.device).exp()
            focal_candidates_lin = torch.linspace(self.focal_min, self.focal_max, self.focal_num_candidates, device=focal_length_probs.device)
            focal_candidates = .75 * focal_candidates_log + .25 * focal_candidates_lin
            focal_candidates = focal_candidates.view(1, -1).expand(focal_length_probs.shape[0], -1)

            focal_length = torch.sum(focal_length_probs * focal_candidates, dim=-1)
        
        elif self.focal_parameterization == "candidates":
            focal_enc = focal_enc * 0.01

            focal_length_probs = F.softmax(focal_enc, dim=-1)

            focal_candidates = torch.linspace(self.focal_min, self.focal_max, self.focal_num_candidates, device=focal_length_probs.device)
            focal_candidates = focal_candidates.view(1, -1).expand(focal_length_probs.shape[0], -1)

            focal_length = torch.sum(focal_length_probs * focal_candidates, dim=-1)

        else:
            raise NotImplementedError("Focal length parameterization not implemented")

        return focal_length, focal_length_probs, focal_candidates

    def stabilize_focal_logits(self, focal_enc):
        focal_enc_center = self.focal_enc_center * (1 - self.center_focal_logits_rate) + focal_enc.mean(dim=0, keepdim=True) * self.center_focal_logits_rate

        focal_enc = focal_enc - focal_enc_center

        focal_enc = focal_enc * self.sharp_focal_logits_temp

        return focal_enc