import logging
import math
from typing import Any, Callable
from dotdict import dotdict
import ignite.distributed as idist

import torch
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine
from torchvision.utils import make_grid
from torchvision.utils import flow_to_image


from anycam.visualization.common import color_tensor

# TODO: configure logger somewhere else
logger = logging.getLogger("Visualization")



def get_input_imgs(data) -> torch.Tensor | None:
    if "imgs" in data and type(data["imgs"]) == list:
        return torch.stack(data["imgs"], dim=1).detach()[0] * 0.5 + 0.5
    elif "imgs" in data:
        return data["imgs"].detach()[0]
    logger.warning(
        "No images found in model output. Not creating a input image visualization."
    )
    return None




def get_depth(data) -> torch.Tensor | None:
    if "pred_depths" in data:
        depth = data["pred_depths"].detach()[0]
        z_near = data["z_near"]
        z_far = data["z_far"]

        depth = (1 / depth - 1 / z_far) / (1 / z_near - 1 / z_far)

        return color_tensor(depth.squeeze(1).clamp(0, 1), cmap="plasma").permute(
            0, 3, 1, 2
        )

    logger.warning(
        "No reconstructed depth found in model output. Not creating a depth visualization."
    )
    return None


def get_uncertainty(data) -> torch.Tensor | None:
    if "uncertainties" in data:
        uncert = data["uncertainties"][0, :, 0, :, :].detach()

        return color_tensor(uncert, cmap="plasma", norm=True).permute(0, 3, 1, 2)

    logger.warning(
        "No uncertainty found in model output. Not creating a uncertainty visualization."
    )
    return None


def get_rendered_flow(data) -> torch.Tensor | None:
    if "induced_flow" in data:
        flow = data["induced_flow"].detach()[0]

        h, w = flow.shape[-2:]
        nv = flow.shape[0]

        flow = flow.permute(0, 2, 3, 1).reshape(nv, h, w, 2).to(torch.float32)

        flow = torch.cat((flow[:, :, :, 0:1] / 2 * w , flow[:, :, :, 1:2] / 2 * h), dim=-1).permute(0, 3, 1, 2)

        flow_imgs = []
        for i in range(nv):
            flow_imgs.append(flow_to_image(flow[i].cpu().squeeze().clamp(-1000, 1000)).float() / 255)

        flow_imgs = torch.stack(flow_imgs, dim=0)
        return flow_imgs
    
    logger.warning(
        "No rendered flows found in model output. Not creating a rendered_flow visualization."
    )
    return None


def get_gt_flow(data) -> torch.Tensor | None:
    if "images_ip" in data:
        flow = data["images_ip"].detach()[0][:, 3:5]

        h, w = flow.shape[-2:]
        nv = flow.shape[0]

        flow = flow.permute(0, 2, 3, 1).reshape(nv, h, w, 2)

        flow = torch.cat((flow[:, :, :, 0:1] / 2 * w , flow[:, :, :, 1:2] / 2 * h), dim=-1).permute(0, 3, 1, 2)

        flow_imgs = []
        for i in range(nv):
            flow_imgs.append(flow_to_image(flow[i].cpu().squeeze()).float() / 255)

        flow_imgs = torch.stack(flow_imgs, dim=0)
        return flow_imgs
        
    logger.warning(
        "No gt flows found in model output. Not creating a rendered_flow visualization."
    )
    return None


def tb_visualize(model, dataset, config: dict[str, Any] | None = None):
    if config is None:
        vis_fns: dict[str, Callable[[Any], torch.Tensor | None]] = {
            "input_imgs": get_input_imgs,
            "depth": get_depth,
            "rendered_flow": get_rendered_flow,
            "gt_flow": get_gt_flow,
            "uncertainty": get_uncertainty,
        }
    else:
        # TODO: inform user about not found functions
        vis_fns = {
            name: globals()[f"get_{name}"]
            for name, _ in config.items()
            if [globals().get(f"get_{name}", None)]
        }

    def _visualize(engine: Engine, tb_logger: TensorboardLogger, step: int, tag: str):
        data = engine.state.output["output"]

        writer = tb_logger.writer
        for name, vis_fn in vis_fns.items():
            output = vis_fn(data)
            if output is not None:
                if name == "profiles":
                    grid = make_grid(output)
                else:
                    grid = make_grid(output, nrow=int(math.sqrt(output.shape[0])))
                writer.add_image(f"{tag}/{name}", grid.cpu(), global_step=step)

    return _visualize
