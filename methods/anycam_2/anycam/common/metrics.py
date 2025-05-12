import logging
import sys
import math
from typing import Callable, Mapping

import skimage.metrics as sk_metrics
import torch
import torch.nn.functional as F
from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce


logger = logging.getLogger("metrics")


def median_scaling(
    depth_gt: torch.Tensor,
    depth_pred: torch.Tensor,
):
    # TODO: This implementation only works for batch size 1
    # mask = depth_gt > 0
    # depth_gt[mask] = torch.nan
    # depth_pred[mask] = torch.nan

    scaling = torch.median(depth_gt) / torch.median(depth_pred)
    depth_pred = scaling * depth_pred
    return depth_pred


def leastsquares_scaling(
    depth_gt: torch.Tensor,
    depth_pred: torch.Tensor,
):
    inv_depth_gt = 1 / depth_gt
    inv_depth = 1 / depth_pred

    n, = depth_gt.shape

    inv_depth = torch.stack((inv_depth, torch.ones_like(inv_depth)), dim=-1)

    x = torch.linalg.lstsq(inv_depth.view(n, 2), inv_depth_gt.view(n, 1)).solution

    depth_pred = 1 / (inv_depth @ x).squeeze()

    # print(x, depth_pred.mean(), depth_gt.mean())

    if torch.any(depth_pred <= 0):
        logger.warning("Least squares scaling produced negative depth values")

    return depth_pred


def median_scaling_shifting(
    depth_gt: torch.Tensor,
    depth_pred: torch.Tensor,
):
    gt_median = torch.median(depth_gt)
    gt_scale = torch.mean((gt_median - depth_gt).abs())

    pred_median = torch.median(depth_pred)
    pred_scale = torch.mean((pred_median - depth_pred).abs())

    depth_pred = (depth_pred - pred_median) / pred_scale * gt_scale + gt_median
    return depth_pred


def l2_scaling(
    depth_gt: torch.Tensor,
    depth_pred: torch.Tensor,
):
    # TODO: ensure this works for any batch size
    mask = depth_gt > 0
    depth_pred = depth_pred
    depth_gt_ = depth_gt[mask]
    depth_pred_ = depth_pred[mask]
    depth_pred_ = torch.stack((depth_pred_, torch.ones_like(depth_pred_)), dim=-1)
    x = torch.linalg.lstsq(
        depth_pred_.to(torch.float32), depth_gt_.unsqueeze(-1).to(torch.float32)
    ).solution.squeeze()
    depth_pred = depth_pred * x[0] + x[1]
    return depth_pred


def compute_depth_metrics(
    depth_gt: torch.Tensor,
    depth_pred: torch.Tensor,
    scaling_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None,
    inverse_depth: bool = False,
    min_depth: float = 1e-3,
    max_depth: float = 75,
):
    w_gt, h_gt = depth_gt.shape[-2:]
    w_pred, h_pred = depth_pred.shape[-2:]

    if w_gt != w_pred or h_gt != h_pred:
        depth_pred = 1 / F.interpolate(
            1 / depth_pred, (w_gt, h_gt), mode="bilinear", align_corners=False
        )

    mask = depth_gt != 0
    mask = mask | ((depth_gt > min_depth) & (depth_gt < max_depth))

    # mask = mask & (depth_pred < 998)

    count = 0

    abs_rel = []
    sq_rel = []
    rmse = []
    rmse_log = []
    a1 = []
    a2 = []
    a3 = []

    for d_gt, d_pred, m in zip(depth_gt, depth_pred, mask):
        d_gt = d_gt[m]
        d_pred = d_pred[m]

        if scaling_fn:
            d_pred = scaling_fn(d_gt, d_pred)

        d_pred = torch.clamp(d_pred, min_depth, max_depth)

        if inverse_depth:
            d_gt = 1 / d_gt
            d_pred = 1 / d_pred

        max_ratio = torch.maximum((d_gt / d_pred), (d_pred / d_gt))
        a_scores = {}
        for name, thresh in {"a1": 1.25, "a2": 1.25**2, "a3": 1.25**3}.items():
            within_thresh = (max_ratio < thresh).to(torch.float)
            a_scores[name] = within_thresh.float().mean()

        _abs_rel = (torch.abs(d_gt - d_pred) / d_gt).mean()
        _sq_rel = ((d_gt - d_pred) ** 2 / d_gt).mean()
        _rmse = ((d_gt - d_pred) ** 2).mean() ** 0.5
        _rmse_log = ((torch.log(d_gt) - torch.log(d_pred)) ** 2).mean() ** 0.5

        abs_rel.append(_abs_rel)
        sq_rel.append(_sq_rel)
        rmse.append(_rmse)
        rmse_log.append(_rmse_log)
        a1.append(a_scores["a1"])
        a2.append(a_scores["a2"])
        a3.append(a_scores["a3"])

        count += 1

    abs_rel = torch.tensor(abs_rel)
    sq_rel = torch.tensor(sq_rel)
    rmse = torch.tensor(rmse)
    rmse_log = torch.tensor(rmse_log)
    a1 = torch.tensor(a1)
    a2 = torch.tensor(a2)
    a3 = torch.tensor(a3)

    metrics_dict = {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "a1": a1,
        "a2": a2,
        "a3": a3,
    }
    return metrics_dict


def compute_occ_metrics(
    occupancy_pred: torch.Tensor, occupancy_gt: torch.Tensor, is_visible: torch.Tensor
):
    # Only not visible points can be occupied
    occupancy_gt &= ~is_visible

    is_occupied_acc = (occupancy_pred == occupancy_gt).float().mean().item()
    is_occupied_prec = occupancy_gt[occupancy_pred].float().mean().item()
    is_occupied_rec = occupancy_pred[occupancy_gt].float().mean().item()

    not_occupied_not_visible_ratio = (
        ((~occupancy_gt) & (~is_visible)).float().mean().item()
    )

    total_ie = ((~occupancy_gt) & (~is_visible)).float().sum().item()

    ie_acc = (occupancy_pred == occupancy_gt)[(~is_visible)].float().mean().item()
    ie_prec = (~occupancy_gt)[(~occupancy_pred) & (~is_visible)].float().mean()
    ie_rec = (~occupancy_pred)[(~occupancy_gt) & (~is_visible)].float().mean()
    total_no_nop_nv = (
        ((~occupancy_gt) & (~occupancy_pred))[(~is_visible) & (~occupancy_gt)]
        .float()
        .sum()
    )

    return {
        "o_acc": is_occupied_acc,
        "o_rec": is_occupied_rec,
        "o_prec": is_occupied_prec,
        "ie_acc": ie_acc,
        "ie_rec": ie_rec,
        "ie_prec": ie_prec,
        "ie_r": not_occupied_not_visible_ratio,
        "t_ie": total_ie,
        "t_no_nop_nv": total_no_nop_nv,
    }


def compute_nvs_metrics(data, lpips):
    # TODO: This is only correct for batchsize 1!
    # Following tucker et al. and others, we crop 5% on all sides

    # idx of stereo frame (the target frame is always the "stereo" frame).
    sf_id = data["rgb_gt"].shape[1] // 2

    imgs_gt = data["rgb_gt"][:1, sf_id : sf_id + 1]
    imgs_pred = data["fine"][0]["rgb"][:1, sf_id : sf_id + 1]

    imgs_gt = imgs_gt.squeeze(0).permute(0, 3, 1, 2)
    imgs_pred = imgs_pred.squeeze(0).squeeze(-2).permute(0, 3, 1, 2)

    n, c, h, w = imgs_gt.shape
    y0 = int(math.ceil(0.05 * h))
    y1 = int(math.floor(0.95 * h))
    x0 = int(math.ceil(0.05 * w))
    x1 = int(math.floor(0.95 * w))

    imgs_gt = imgs_gt[:, :, y0:y1, x0:x1]
    imgs_pred = imgs_pred[:, :, y0:y1, x0:x1]

    imgs_gt_np = imgs_gt.detach().squeeze().permute(1, 2, 0).cpu().numpy()
    imgs_pred_np = imgs_pred.detach().squeeze().permute(1, 2, 0).cpu().numpy()

    ssim_score = sk_metrics.structural_similarity(
        imgs_pred_np, imgs_gt_np, multichannel=True, data_range=1, channel_axis=-1
    )
    psnr_score = sk_metrics.peak_signal_noise_ratio(
        imgs_pred_np, imgs_gt_np, data_range=1
    )
    lpips_score = lpips(imgs_pred, imgs_gt, normalize=False).mean()

    metrics_dict = {
        "ssim": torch.tensor([ssim_score], device=imgs_gt.device),
        "psnr": torch.tensor([psnr_score], device=imgs_gt.device),
        "lpips": torch.tensor([lpips_score], device=imgs_gt.device),
    }
    return metrics_dict


# TODO: seperate files


class MeanMetric(Metric):
    def __init__(self, output_transform=lambda x: x["output"], device="cpu"):
        self._sum = torch.tensor(0, device=self._device, dtype=torch.float32)
        self._num_examples = 0
        self.required_output_keys = ()
        super(MeanMetric, self).__init__(
            output_transform=output_transform, device=device
        )

    @reinit__is_reduced
    def reset(self):
        self._sum = torch.tensor(0, device=self._device, dtype=torch.float32)
        self._num_examples = 0
        super(MeanMetric, self).reset()

    @reinit__is_reduced
    def update(self, value):
        if torch.any(torch.isnan(torch.tensor(value))):
            return
        self._sum += value
        self._num_examples += 1

    @sync_all_reduce("_num_examples:SUM", "_sum:SUM")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                "CustomAccuracy must have at least one example before it can be computed."
            )
        return self._sum.item() / self._num_examples

    @torch.no_grad()
    def iteration_completed(self, engine: Engine) -> None:
        output = self._output_transform(
            engine.state.output
        )  ## engine.state.output.keys() == dict_keys(['output', 'loss_dict', 'timings_dict', 'metrics_dict'])
        self.update(output)


class DictMeanMetric(Metric):
    def __init__(self, name: str, output_transform=lambda x: x["output"], device="cpu"):
        self._name = name
        self._sums: dict[str, torch.Tensor] = {}
        self._num_examples = 0
        self._num_invalids: dict[str, torch.Tensor] = {}
        self.required_output_keys = ()
        super(DictMeanMetric, self).__init__(
            output_transform=output_transform, device=device
        )

    @reinit__is_reduced
    def reset(self):
        self._sums = {}
        self._num_examples = 0
        self._num_invalids = {}
        super(DictMeanMetric, self).reset()

    @reinit__is_reduced
    def update(self, value):
        num_examples = None
        for key, metric in value.items():
            if not key in self._sums:
                self._sums[key] = torch.tensor(
                    0, device=self._device, dtype=torch.float32
                )
                self._num_invalids[key] = torch.tensor(
                    0, device=self._device, dtype=torch.float32
                )
            if torch.any(torch.isnan(metric)):
                # TODO: integrate into logging
                # print(f"Warining: Metric {self._name}/{key} has a nan value")

                mask = ~torch.isnan(metric)

                self._num_invalids[key] += (~mask).to(torch.float32).sum().to(self._device)

                if torch.all(torch.isnan(metric)):
                    continue

                metric = metric[mask]
            self._sums[key] += metric.sum().to(self._device)
            # TODO: check if this works with batches
            if num_examples is None:
                num_examples = metric.shape[0]
        self._num_examples += num_examples

    @sync_all_reduce("_num_examples:SUM", "_sum:SUM")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                "CustomAccuracy must have at least one example before it can be computed."
            )
        for key, num_invalid in self._num_invalids.items():
            if num_invalid > 0:
                logger.info(
                    f"Warning: Metric {self._name}/{key} has {int(num_invalid)} invalid values."
                )
        return {
            f"{self._name}_{key}": metric.item() / self._num_examples
            for key, metric in self._sums.items()
        }

    @torch.no_grad()
    def iteration_completed(self, engine: Engine) -> None:
        output = self._output_transform(engine.state.output["output"])
        self.update(output)

    def completed(self, engine: Engine, name: str) -> None:
        """Helper method to compute metric's value and put into the engine. It is automatically attached to the
        `engine` with :meth:`~ignite.metrics.metric.Metric.attach`. If metrics' value is torch tensor, it is
        explicitly sent to CPU device.

        Args:
            engine: the engine to which the metric must be attached
            name: the name of the metric used as key in dict `engine.state.metrics`

        .. changes from default implementation:
            don't add whole result dict to engine state, but only the values

        """
        result = self.compute()
        if isinstance(result, Mapping):
            if name in result.keys():
                raise ValueError(
                    f"Argument name '{name}' is conflicting with mapping keys: {list(result.keys())}"
                )

            for key, value in result.items():
                engine.state.metrics[key] = value
        else:
            if isinstance(result, torch.Tensor):
                if len(result.size()) == 0:
                    result = result.item()
                elif "cpu" not in result.device.type:
                    result = result.cpu()

            engine.state.metrics[name] = result


class FG_ARI(Metric):
    def __init__(self, output_transform=lambda x: x["output"], device="cpu"):
        self._sum_fg_aris = torch.tensor(0, device=self._device, dtype=torch.float32)
        self._num_examples = 0
        self.required_output_keys = ()
        super(FG_ARI, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._sum_fg_aris = torch.tensor(0, device=self._device, dtype=torch.float32)
        self._num_examples = 0
        super(FG_ARI, self).reset()

    @reinit__is_reduced
    def update(self, data):
        true_masks = data["segs"]  # fc [n, h, w]
        pred_masks = data["slot_masks"]  # n, fc, sc, h, w

        n, fc, sc, h, w = pred_masks.shape

        true_masks = [
            F.interpolate(tm.to(float).unsqueeze(1), (h, w), mode="nearest")
            .squeeze(1)
            .to(int)
            for tm in true_masks
        ]

        for i in range(n):
            for f in range(fc):
                true_mask = true_masks[f][i]
                pred_mask = pred_masks[i, f]

                true_mask = true_mask.view(-1)
                pred_mask = pred_mask.view(sc, -1)

                if torch.max(true_mask) == 0:
                    continue

                foreground = true_mask > 0
                true_mask = true_mask[foreground]
                pred_mask = pred_mask[:, foreground].permute(1, 0)

                true_mask = F.one_hot(true_mask)

                # Filter out empty true groups
                not_empty = torch.any(true_mask, dim=0)
                true_mask = true_mask[:, not_empty]

                # Filter out empty predicted groups
                not_empty = torch.any(pred_mask, dim=0)
                pred_mask = pred_mask[:, not_empty]

                true_mask.unsqueeze_(0)
                pred_mask.unsqueeze_(0)

                _, n_points, n_true_groups = true_mask.shape
                n_pred_groups = pred_mask.shape[-1]
                if n_points <= n_true_groups and n_points <= n_pred_groups:
                    print(
                        "adjusted_rand_index requires n_groups < n_points.",
                        file=sys.stderr,
                    )
                    continue

                true_group_ids = torch.argmax(true_mask, -1)
                pred_group_ids = torch.argmax(pred_mask, -1)
                true_mask_oh = true_mask.to(torch.float32)
                pred_mask_oh = F.one_hot(pred_group_ids, n_pred_groups).to(
                    torch.float32
                )

                n_points = torch.sum(true_mask_oh, dim=[1, 2]).to(torch.float32)

                nij = torch.einsum("bji,bjk->bki", pred_mask_oh, true_mask_oh)
                a = torch.sum(nij, dim=1)
                b = torch.sum(nij, dim=2)

                rindex = torch.sum(nij * (nij - 1), dim=[1, 2])
                aindex = torch.sum(a * (a - 1), dim=1)
                bindex = torch.sum(b * (b - 1), dim=1)
                expected_rindex = aindex * bindex / (n_points * (n_points - 1))
                max_rindex = (aindex + bindex) / 2
                ari = (rindex - expected_rindex) / (
                    max_rindex - expected_rindex + 0.000000000001
                )

                _all_equal = lambda values: torch.all(
                    torch.eq(values, values[..., :1]), dim=-1
                )
                both_single_cluster = torch.logical_and(
                    _all_equal(true_group_ids), _all_equal(pred_group_ids)
                )

                self._sum_fg_aris += torch.where(
                    both_single_cluster, torch.ones_like(ari), ari
                ).squeeze()
                self._num_examples += 1

    @sync_all_reduce("_num_examples:SUM", "_sum_fg_aris:SUM")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                "CustomAccuracy must have at least one example before it can be computed."
            )
        return self._sum_fg_aris.item() / self._num_examples

    @torch.no_grad()
    def iteration_completed(self, engine: Engine) -> None:
        output = self._output_transform(engine.state.output)
        self.update(output)
