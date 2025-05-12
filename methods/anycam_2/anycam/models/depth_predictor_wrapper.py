import math
import time
from typing import Optional, Tuple, Union
import warnings
from einops import rearrange
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from transformers.models.depth_anything.modeling_depth_anything import DepthEstimatorOutput
import torch
import numpy as np

from torch import nn
import torch.nn.functional as F
from math import ceil


class DepthAnythingWrapper(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
        self,
    ):
        super().__init__()

        self.image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        self.model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")


    def da_forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], DepthEstimatorOutput]:
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions

        outputs = self.model.backbone.forward_with_filtered_kwargs(
            pixel_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions
        )
        hidden_states = outputs.feature_maps

        _, _, height, width = pixel_values.shape
        patch_size = self.model.config.patch_size
        patch_height = height // patch_size
        patch_width = width // patch_size

        hidden_states = self.model.neck(hidden_states, patch_height, patch_width)

        predicted_depth = self.model.head(hidden_states, patch_height, patch_width)

        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not implemented yet")

        if not return_dict:
            if output_hidden_states:
                output = (predicted_depth,) + outputs[1:]
            else:
                output = (predicted_depth,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return DepthEstimatorOutput(
            loss=loss,
            predicted_depth=predicted_depth,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        ), hidden_states[self.model.config.head_in_index]


    def forward(self, x, return_features=False):
        n, c, h, w = x.shape

        x = x - torch.tensor([[0.485, 0.456, 0.406]], device=x.device).view(1, 3, 1, 1)
        x = x / torch.tensor([[0.229, 0.224, 0.225]], device=x.device).view(1, 3, 1, 1)

        if h < w:
            size = (518, math.ceil(w * 518 / h / 14) * 14)
        else:
            size = (math.ceil(h * 518 / w / 14) * 14, 518)

        x = F.interpolate(x, size=size, mode="bicubic", align_corners=True)

        inputs = {
            "pixel_values": x,
        }

        outputs, features = self.da_forward(**inputs)

        predicted_depth = outputs.predicted_depth

        prediction = F.interpolate(
            predicted_depth.unsqueeze(1),
            size=(h, w),
            mode="nearest",
            # align_corners=True,
        )

        if not return_features:
            return [prediction]
        else:
            return [prediction], features

    @classmethod
    def from_conf(cls, conf):
        return cls()


RESOLUTION_LEVELS = 10


def _check_ratio(image_ratio, ratio_bounds):
    ratio_bounds = sorted(ratio_bounds)
    if ratio_bounds is not None and (
        image_ratio < ratio_bounds[0] or image_ratio > ratio_bounds[1]
    ):
        warnings.warn(
            f"Input image ratio ({image_ratio:.3f}) is out of training "
            f"distribution: {ratio_bounds}. This may lead to unexpected results. "
            f"Consider resizing/padding the image to match the training distribution."
        )


def _check_resolution(shape_constraints, resolution_level):
    if resolution_level is None:
        resolution_level = RESOLUTION_LEVELS
    pixel_bounds = sorted(shape_constraints["pixels_bounds_ori"])
    pixel_range = pixel_bounds[-1] - pixel_bounds[0]
    clipped_resolution_level = min(max(resolution_level, 0), RESOLUTION_LEVELS)
    if clipped_resolution_level != resolution_level:
        shape_constraints["pixels_bounds"] = [
            pixel_bounds[0]
            + ceil(pixel_range * clipped_resolution_level / RESOLUTION_LEVELS),
            pixel_bounds[0]
            + ceil(pixel_range * clipped_resolution_level / RESOLUTION_LEVELS),
        ]
    return shape_constraints


def _get_closes_num_pixels(image_shape, pixels_bounds):
    h, w = image_shape
    num_pixels = h * w
    pixels_bounds = sorted(pixels_bounds)
    num_pixels = max(min(num_pixels, pixels_bounds[1]), pixels_bounds[0])
    return num_pixels


def _shapes(image_shape, shape_constraints):
    h, w = image_shape
    image_ratio = w / h
    _check_ratio(image_ratio, shape_constraints["ratio_bounds"])
    num_pixels = _get_closes_num_pixels(
        (h / shape_constraints["patch_size"], w / shape_constraints["patch_size"]),
        shape_constraints["pixels_bounds"],
    )
    h = ceil((num_pixels / image_ratio) ** 0.5 - 0.5)
    w = ceil(h * image_ratio - 0.5)
    ratio = h / image_shape[0] * shape_constraints["patch_size"]
    return (
        h * shape_constraints["patch_size"],
        w * shape_constraints["patch_size"],
    ), ratio


def _preprocess(rgbs, intrinsics, shapes, ratio):
    rgbs = F.interpolate(rgbs, size=shapes, mode="bilinear", antialias=True)
    if intrinsics is not None:
        intrinsics = intrinsics.clone()
        intrinsics[:, 0, 0] = intrinsics[:, 0, 0] * ratio
        intrinsics[:, 1, 1] = intrinsics[:, 1, 1] * ratio
        intrinsics[:, 0, 2] = intrinsics[:, 0, 2] * ratio
        intrinsics[:, 1, 2] = intrinsics[:, 1, 2] * ratio
        return rgbs, intrinsics
    return rgbs, None


def _postprocess(outs, ratio, original_shapes, mode="nearest-exact"):
    outs["depth"] = F.interpolate(outs["depth"], size=original_shapes, mode=mode)
    outs["confidence"] = F.interpolate(
        outs["confidence"], size=original_shapes, mode="bilinear", antialias=True
    )
    outs["K"][:, 0, 0] = outs["K"][:, 0, 0] / ratio
    outs["K"][:, 1, 1] = outs["K"][:, 1, 1] / ratio
    outs["K"][:, 0, 2] = outs["K"][:, 0, 2] / ratio
    outs["K"][:, 1, 2] = outs["K"][:, 1, 2] / ratio
    return outs



class UniDepthV2Wrapper(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
        self,
        conf,
    ):
        super().__init__()

        version = conf.get("version", "v2")
        backbone = conf.get("backbone", "vits14")
        self.scaling = conf.get("scaling", 0.1)

        self.model = torch.hub.load("Brummi/UniDepth:stable", "UniDepth", version=version, backbone=backbone, pretrained=True, trust_repo=True, force_reload=False)

        self.pixel_encoder = self.model.pixel_encoder
        self.pixel_decoder = self.model.pixel_decoder

    def forward(self, rgbs, return_features=False):
        n, c, h, w = rgbs.shape

        start_time = time.time()

        rgbs = rgbs - torch.tensor([[0.485, 0.456, 0.406]], device=rgbs.device).view(1, 3, 1, 1)
        rgbs = rgbs / torch.tensor([[0.229, 0.224, 0.225]], device=rgbs.device).view(1, 3, 1, 1)

        n, c, H, W = rgbs.shape


        shape_constraints = _check_resolution(self.model.shape_constraints, self.model.resolution_level)

        # get image shape
        (h, w), ratio = _shapes((H, W), shape_constraints)
        rgbs, gt_intrinsics = _preprocess(
            rgbs,
            None,
            (h, w),
            ratio,
        )

        # run encoder
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
            features, tokens = self.pixel_encoder(rgbs)

            cls_tokens = [x.contiguous() for x in tokens]
            features = [
                self.model.stacking_fn(features[i:j]).contiguous()
                for i, j in self.model.slices_encoder_range
            ]
            tokens = [
                self.model.stacking_fn(tokens[i:j]).contiguous()
                for i, j in self.model.slices_encoder_range
            ]
            global_tokens = [cls_tokens[i] for i in [-2, -1]]
            camera_tokens = [cls_tokens[i] for i in [-3, -2, -1]] + [tokens[-2]]

            # get data fro decoder and adapt to given camera
            inputs = {}
            inputs["features"] = features
            inputs["tokens"] = tokens
            inputs["global_tokens"] = global_tokens
            inputs["camera_tokens"] = camera_tokens
            inputs["image"] = rgbs

            outs = self.pixel_decoder(inputs, {})
        # undo the reshaping and get original image size (slow)
        outs = _postprocess(outs, ratio, (H, W), mode=self.model.interpolation_mode)
        depth = outs["depth"]

        depth = depth * self.scaling

        depth = 1 / depth

        if not return_features:
            return [depth]
        else:
            return [depth], torch.zeros_like(depth)

    @classmethod
    def from_conf(cls, conf):
        return cls(conf)

class Metric3DV2Wrapper(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
        self,
        conf,
    ):
        super().__init__()

        self.variant = conf.get("variant", "metric3d_vit_large")
        self.scaling = conf.get("scaling", 0.1)

        
        self.model = torch.hub.load('yvanyin/metric3d', self.variant, pretrain=True)

    def forward(self, rgbs, return_features=False):
        n, c, h, w = rgbs.shape

        start_time = time.time()

        rgbs = rgbs - torch.tensor([[0.485, 0.456, 0.406]], device=rgbs.device).view(1, 3, 1, 1)
        rgbs = rgbs / torch.tensor([[0.229, 0.224, 0.225]], device=rgbs.device).view(1, 3, 1, 1)

        n, c, H, W = rgbs.shape

        depth, confidence, output_dict = self.model.inference({'input': rgbs})

        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=False)

        depth = depth * self.scaling

        depth = 1 / depth

        if not return_features:
            return [depth]
        else:
            return [depth], torch.zeros_like(depth)

    @classmethod
    def from_conf(cls, conf):
        return cls(conf)
