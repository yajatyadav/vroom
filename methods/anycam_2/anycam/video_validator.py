from pathlib import Path
import time
from ignite.engine import Engine, Events, EventsList
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.contrib.engines import common
import ignite.distributed as idist
import numpy as np
from omegaconf import OmegaConf

from torch.nn.parallel import DataParallel, DistributedDataParallel

from anycam.common.logging import event_list_from_config, global_step_fn
from anycam.training.handlers import MetricLoggingHandler, add_time_handlers
from anycam.scripts.evaluate_trajectories import run_eval
from anycam.scripts.fit_video import fit_video_wrapper 


class VideoEngine(Engine):
    def __init__(self, *args, **kwargs):
        super().__init__(lambda engine, data: {})

        self.model_fn = kwargs["model_fn"]

    def run(self, data_loader, *args, **kwargs):
        start_time = time.time()

        avg_results, results, failed_sequences = run_eval(
            model=self.model_fn,
            dataloader=data_loader,
            with_rerun=False,
            mode="global",
        )

        output = {
            "ate": avg_results["ape_mean"],
            "rte": avg_results["rte_mean"],
            "rre": avg_results["rre_mean"],
            "focal_error": avg_results["mean_fy_error"],
            "focal_error_rel": avg_results["rel_fy_error"],
        }

        self.state.output = output
        self.state.metrics = output
        self.state.output["metrics_dict"] = {}
        self.state.output["timings_dict"] = {}

        final_time = time.time() - start_time
        self.state.times["COMPLETED"] = final_time

        self.fire_event(Events.EPOCH_COMPLETED)
        
        return self.state


def video_validator(
    model,
    tag: str,
    validation_config,
    with_amp,
    device,
    tb_logger,
    trainer,
    config,
):
    fit_video_config_path = validation_config["fit_video_config"]

    fit_video_config = OmegaConf.load(fit_video_config_path)

    if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
        model = model.module

    def run_anycam(imgs, proj, seq_name):
        model.eval()
        imgs = [img.astype(np.float32) / 255 for img in imgs]
        poses, projs = fit_video_wrapper(fit_video_config, model, None, imgs, device)
        return poses, projs

    validator = VideoEngine(
        model_fn=run_anycam
    )

    event_triggers = event_list_from_config(validation_config["events"])

    add_time_handlers(validator)

    # ADD LOGGING HANDLER
    # TODO: split up handlers
    tb_logger.attach(
        validator,
        MetricLoggingHandler(
            tag,
            log_loss=False,
            log_timings=False,
            global_step_transform=global_step_fn(
                trainer, validation_config["global_step"]
            ),
        ),
        Events.EPOCH_COMPLETED,
    )

    if "save_best" in validation_config:
        # Store 2 best models by validation accuracy starting from num_epochs / 2:
        save_best_config = validation_config["save_best"]
        metric_name = save_best_config["metric"]
        sign = save_best_config.get("sign", 1.0)

        best_model_handler = Checkpoint(
            {"model": model},
            # NOTE: fixes a problem with log_dir or logdir
            DiskSaver(Path(config["output"]["path"]), require_empty=False),
            # DiskSaver(tb_logger.writer.log_dir, require_empty=False),
            filename_prefix=f"{metric_name}_best",
            n_saved=5,
            global_step_transform=global_step_from_engine(trainer),
            score_name=metric_name,
            score_function=Checkpoint.get_default_score_fn(
                metric_name, score_sign=sign
            ),
        )

        validator.add_event_handler(
            Events.COMPLETED(
                lambda *_: (
                    (trainer.state.epoch > trainer.state.max_epochs // 2)
                    if trainer.state.max_epochs
                    else False
                )
            ),
            best_model_handler,
        )

    if idist.get_rank() == 0 and (not validation_config.get("with_clearml", False)):
        common.ProgressBar(desc=f"Evaluation ({tag})", persist=False).attach(
            validator
        )

    return validator, event_triggers
