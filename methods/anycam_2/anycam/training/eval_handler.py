from typing import Optional
# import lpips
import torch
import torch.nn as nn

from anycam.loss.metric import camera_to_rel_deg


def create_pose_eval(
    model: nn.Module,
    min_translation: Optional[float] = None,
):
    def _compute_pose_metrics(
        data,
    ):
        proc_pose = data["proc_poses"][:, 0]
        gt_rel_pose = torch.inverse(data["poses"][:, 1]) @ data["poses"][:, 0]
        n = gt_rel_pose.shape[0]
        proc_pose = proc_pose[:n]
        rre, rte = camera_to_rel_deg(gt_rel_pose, proc_pose)

        if min_translation is None:
            is_static = torch.norm(gt_rel_pose[:, :3, 3], dim=-1) < min_translation
            rte[is_static] = float("nan")

        metrics_dict = {
            "rre": rre,
            "rte": rte,
        }
        return metrics_dict

    return _compute_pose_metrics


def make_eval_fn(
    model: nn.Module,
    conf,
):
    eval_type = conf["type"]

    eval_fn = globals().get(f"create_{eval_type}_eval", None)
    if eval_fn:
        if conf.get("args", None):
            return eval_fn(model, **conf["args"])
        else:
            return eval_fn(model)
    else:
        return None
