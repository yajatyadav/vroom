import logging
from collections import defaultdict
from dataclasses import field, dataclass

import math
import numpy as np

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import collections
from torch import Tensor
from itertools import repeat

from minipytorch3d.harmonic_embedding import HarmonicEmbedding


logger = logging.getLogger(__name__)


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


Depth_Anything_V2_Small_hf = {
  "_commit_hash": None,
  "architectures": [
    "DepthAnythingForDepthEstimation"
  ],
  "backbone": None,
  "backbone_config": {
    "architectures": [
      "Dinov2Model"
    ],
    "hidden_size": 384,
    "image_size": 518,
    "model_type": "dinov2",
    "num_attention_heads": 6,
    "out_features": [
      "stage3",
      "stage6",
      "stage9",
      "stage12"
    ],
    "out_indices": [
      3,
      6,
      9,
      12
    ],
    "patch_size": 14,
    "reshape_hidden_states": False,
  },
  "use_pretrained_backbone": True,
  "fusion_hidden_size": 64,
  "head_hidden_size": 32,
  "head_in_index": -1,
  "initializer_range": 0.02,
  "model_type": "depth_anything",
  "neck_hidden_sizes": [
    48,
    96,
    192,
    384
  ],
  "patch_size": 14,
  "reassemble_factors": [
    4,
    2,
    1,
    0.5
  ],
  "reassemble_hidden_size": 384,
  "torch_dtype": "float32",
  "transformers_version": None,
  "use_pretrained_backbone": False
}

# Adapted from the DepthAnything architecture

class AnyCamPoseTokenReassembleStage(nn.Module):
    def __init__(self, in_chn, out_chn: int, bottleneck_chns: List[int]):
        super().__init__()

        self.projections = nn.ModuleList([])

        for c in bottleneck_chns:
            self.projections.append(
                nn.Sequential(
                    nn.Linear(in_chn, c),
                    nn.Linear(c, out_chn),
                )
            )

    def forward(self, pose_tokens):
        pose_tokens = [proj(token) for token, proj in zip(pose_tokens, self.projections)]

        return pose_tokens
    

class AnyCamPoseTokenFusionPreAct(nn.Module):
    """
    ResidualConvUnit, pre-activate residual unit.

    Args:
        config (`[DepthAnythingConfig]`):
            Model configuration class defining the model architecture.
    """

    def __init__(self, chn):
        super().__init__()

        self.activation1 = nn.ReLU()
        self.convolution1 = nn.Linear(chn, chn)
        self.activation2 = nn.ReLU()
        self.convolution2 = nn.Linear(chn, chn)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        residual = hidden_state
        hidden_state = self.activation1(hidden_state)
        hidden_state = self.convolution1(hidden_state)
        hidden_state = self.activation2(hidden_state)
        hidden_state = self.convolution2(hidden_state)
        return hidden_state + residual


class AnyCamPoseTokenFusionStage(nn.Module):
    def __init__(self, chn: int, num_featuremaps=4):
        super().__init__()

        self.projections = nn.ModuleList([])
        self.residual_blocks = nn.ModuleList([])

        for _ in range(num_featuremaps):
            self.projections.append(nn.Linear(chn, chn))

            residual0 = AnyCamPoseTokenFusionPreAct(chn)
            residual1 = AnyCamPoseTokenFusionPreAct(chn)

            self.residual_blocks.append(nn.ModuleList([residual0, residual1]))

    def forward(self, pose_tokens):
        pose_tokens = pose_tokens[::-1]

        fused_pose_tokens = []
        # first layer only uses the last hidden_state
        fused_pose_token = self.residual_blocks[0][1](pose_tokens[0])
        fused_pose_token = self.projections[0](fused_pose_token)
        fused_pose_tokens.append(fused_pose_token)

        # looping from the last layer to the second
        for idx, (pose_token, residual_block) in enumerate(zip(pose_tokens[1:], self.residual_blocks[1:])):
            pose_token = residual_block[0](pose_token)
            fused_pose_token = fused_pose_token + pose_token
            fused_pose_token = residual_block[1](fused_pose_token)
            fused_pose_token = self.projections[idx](fused_pose_token)

            fused_pose_tokens.append(fused_pose_token)

        return fused_pose_tokens


class AnyCamPoseTokenHead(nn.Module):
    def __init__(self, in_chn: int, out_chn: int):
        super().__init__()

        self.proj0 = nn.Linear(in_chn, in_chn // 2)
        self.activation0 = nn.ReLU()
        self.proj1 = nn.Linear(in_chn // 2, out_chn)
    
    def forward(self, fused_pose_token):
        fused_pose_token = self.proj0(fused_pose_token)
        fused_pose_token = self.activation0(fused_pose_token)
        fused_pose_token = self.proj1(fused_pose_token)

        return fused_pose_token


def pose_scaling_linear(scale=0.01):
    def pose_scaling(x):
        return x * scale
    
    return pose_scaling


def pose_scaling_quadratic(scale=0.1):
    def pose_scaling(x):
        return x ** 2 * scale * torch.sign(x)
    return pose_scaling


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


to_2tuple = _ntuple(2)


class ResidualBlock(nn.Module):
    """
    ResidualBlock: construct a block of two conv layers with residual connections
    """

    def __init__(
        self, in_planes, planes, norm_fn="group", stride=1, kernel_size=3
    ):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=kernel_size,
            padding=1,
            stride=stride,
            padding_mode="zeros",
        )
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=kernel_size,
            padding=1,
            padding_mode="zeros",
        )
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(
                num_groups=num_groups, num_channels=planes
            )
            self.norm2 = nn.GroupNorm(
                num_groups=num_groups, num_channels=planes
            )
            if not stride == 1:
                self.norm3 = nn.GroupNorm(
                    num_groups=num_groups, num_channels=planes
                )

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()
        else:
            raise NotImplementedError

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                self.norm3,
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = (
            partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        )

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class AttnBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        attn_class: Callable[..., nn.Module] = nn.MultiheadAttention,
        mlp_ratio=4.0,
        **block_kwargs
    ):
        """
        Self attention block
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )

        self.attn = attn_class(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
            **block_kwargs
        )

        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, drop=0
        )

    def forward(self, x, mask=None):
        # Prepare the mask for PyTorch's attention (it expects a different format)
        # attn_mask = mask if mask is not None else None
        # Normalize before attention
        x = self.norm1(x)

        # PyTorch's MultiheadAttention returns attn_output, attn_output_weights
        # attn_output, _ = self.attn(x, x, x, attn_mask=attn_mask)

        attn_output, _ = self.attn(x, x, x)

        # Add & Norm
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttnBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        context_dim,
        num_heads=1,
        mlp_ratio=4.0,
        **block_kwargs
    ):
        """
        Cross attention block
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.norm_context = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
            **block_kwargs
        )

        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, drop=0
        )

    def forward(self, x, context, mask=None):
        # Normalize inputs
        x = self.norm1(x)
        context = self.norm_context(context)

        # Apply cross attention
        # Note: nn.MultiheadAttention returns attn_output, attn_output_weights
        attn_output, _ = self.cross_attn(x, context, context, attn_mask=mask)

        # Add & Norm
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x


class PoseEmbedding(nn.Module):
    def __init__(self, target_dim, n_harmonic_functions=10, append_input=True):
        super().__init__()

        self._emb_pose = HarmonicEmbedding(
            n_harmonic_functions=n_harmonic_functions, append_input=append_input
        )

        self.out_dim = self._emb_pose.get_output_dim(target_dim)

    def forward(self, pose_encoding):
        e_pose_encoding = self._emb_pose(pose_encoding)
        return e_pose_encoding