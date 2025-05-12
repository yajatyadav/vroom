import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from omegaconf import ListConfig, OmegaConf

import ignite.distributed as idist
import torch
from torch.utils.data import DataLoader
from ignite.contrib.engines import common
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine, Events, EventsList
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.utils import manual_seed, setup_logger
from torch.amp import autocast, GradScaler

from anycam.common.logging import event_list_from_config, global_step_fn, log_basic_info
from anycam.common.io.configs import save_hydra_config
from anycam.common.io.model import load_checkpoint
from anycam.training.eval_handler import make_eval_fn
from anycam.loss.base_loss import BaseLoss
from anycam.training.handlers import (
    MetricLoggingHandler,
    VisualizationHandler,
    add_time_handlers,
)
from anycam.common.array_operations import to
from anycam.common.metrics import DictMeanMetric, MeanMetric
from anycam.visualization.vis_2d import tb_visualize


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)



def base_training(local_rank, config, get_dataflow, initialize, get_custom_trainer_events=None):
    # ============================================ LOGGING AND OUTPUT SETUP ============================================
    # TODO: figure out rank
    rank = (
        idist.get_rank()
    )  ## rank of the current process within a group of processes: each process could handle a unique subset of the data, based on its rank
    manual_seed(config["seed"] + rank)
    device = idist.device()

    model_name = config["name"]
    logger = setup_logger(
        name=model_name, format="%(levelname)s: %(message)s"
    )  ## default

    output_path = config["output"]["path"]
    if rank == 0:
        unique_id = config["output"].get(
            "unique_id", datetime.now().strftime("%Y%m%d-%H%M%S")
        )

        folder_name = f"{model_name}_backend-{idist.backend()}-{idist.get_world_size()}_{unique_id}"

        output_path = Path(output_path) / folder_name
        if not output_path.exists():
            output_path.mkdir(parents=True)

        config["output"]["path"] = output_path.as_posix()
        logger.info(f"Output path: {config['output']['path']}")

        if "cuda" in device.type:
            config["cuda device name"] = torch.cuda.get_device_name(local_rank)
    log_basic_info(logger, config)
    tb_logger = TensorboardLogger(log_dir=output_path)

    # ================================================== DATASET SETUP =================================================
    # TODO: think about moving the dataset setup to the create validators and create trainer functions
    train_loader, val_loaders = get_dataflow(config)

    if hasattr(train_loader, "dataset"):
        val_loader_lengths = "\n".join(
            [
                f"{name}: {len(val_loader.dataset)}"
                for name, val_loader in val_loaders.items()
                if hasattr(val_loader, "dataset")
            ]
        )
        logger.info(
            f"Dataset lengths:\nTrain: {len(train_loader.dataset)}\n{val_loader_lengths}"
        )
    # config["dataset"]["steps_per_epoch"] = len(train_loader)

    # ============================================= MODEL AND OPTIMIZATION =============================================
    model, optimizer, criterion, lr_scheduler = initialize(config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Create trainer for current task
    trainer = create_trainer(
        model,
        optimizer,
        criterion,
        lr_scheduler,
        train_loader.sampler if hasattr(train_loader, "sampler") else None,
        config,
        logger,
        metrics={},
    )
    if rank == 0:
        tb_logger.attach(
            trainer,
            MetricLoggingHandler("train", optimizer),
            Events.ITERATION_COMPLETED(every=config.get("log_every_iters", 1)),
        )

    # ========================================= EVALUTATION, AND VISUALIZATION =========================================
    validators: dict[str, tuple[Engine, EventsList]] = create_validators(
        config,
        model,
        val_loaders,
        criterion,
        tb_logger,
        trainer,
    )

    # NOTE: not super elegant as val_loaders has to have the same name but should work
    def run_validation(name: str, validator: Engine):
        def _run(engine: Engine):
            epoch = trainer.state.epoch
            for name_vl, val_loader in val_loaders.items():
                if name_vl == name:    
                    state = validator.run(val_loader)
                    log_metrics(logger, epoch, state.times["COMPLETED"], name_vl, state.metrics)

        return _run

    for name, validator in validators.items():
        trainer.add_event_handler(validator[1], run_validation(name, validator[0]))

    # ================================================ SAVE FINAL CONFIG ===============================================
    if rank == 0:
        # Plot config to tensorboard
        config_yaml = OmegaConf.to_yaml(config)
        config_yaml = "".join("\t" + line for line in config_yaml.splitlines(True))
        tb_logger.writer.add_text("config", text_string=config_yaml, global_step=0)
    save_hydra_config(os.path.join(output_path, "training_config.yaml"), config, force=False)

    # ================================================= TRAINING LOOP ==================================================
    # In order to check training resuming we can stop training on a given iteration
    if config["training"].get("stop_iteration", None):

        @trainer.on(Events.ITERATION_STARTED(once=config["training"]["stop_iteration"]))
        def _():
            logger.info(f"Stop training on {trainer.state.iteration} iteration")
            trainer.terminate()

    if get_custom_trainer_events:
        for event, handler in get_custom_trainer_events(config):
            trainer.add_event_handler(event, handler)

    try:  ## train_loader == models.bts.trainer_overfit.DataloaderDummy object
        trainer.run(train_loader, max_epochs=config["training"]["num_epochs"])
    except Exception as e:
        logger.exception("")
        raise e

    if rank == 0:
        tb_logger.close()


def log_metrics(logger, epoch, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(
        f"\nEpoch {epoch} - Evaluation time (seconds): {elapsed:.2f} - {tag} metrics:\n {metrics_output}"
    )


def create_trainer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterions: list[Any],
    lr_scheduler,
    train_sampler,
    config,
    logger,
    metrics={},
):
    device = idist.device()

    # Setup Ignite trainer:
    # - let's define training step
    # - add other common handlers:
    #    - TerminateOnNan,
    #    - handler to setup learning rate scheduling,
    #    - ModelCheckpoint
    #    - RunningAverage` on `train_step` output
    #    - Two progress bars on epochs and optionally on iterations

    with_amp = config["with_amp"]

    scaler = GradScaler(enabled=with_amp)

    def train_step(engine, data: dict):
        if "t__get_item__" in data:
            timing = {"t__get_item__": torch.mean(data["t__get_item__"]).item()}
        else:
            timing = {}

        # Find batch size:
        n = len(next(iter(data.values())))
        data["iteration"] = torch.tensor([engine.state.iteration]).repeat(n)

        _start_time = time.time()

        data = to(data, device)

        timing["t_to_gpu"] = time.time() - _start_time

        model.train()

        _start_time = time.time()

        with autocast(enabled=with_amp, device_type="cuda"):
            data = model(data)

            timing["t_forward"] = time.time() - _start_time

            _start_time = time.time()

            overall_loss = None
            loss_metrics = {}
            for criterion in criterions:
                losses = criterion(data)
                names = criterion.get_loss_metric_names()

                if overall_loss is None:
                    overall_loss = losses[names[0]]
                else:
                    overall_loss = overall_loss + losses[names[0]]
                loss_metrics.update({name: loss for name, loss in losses.items()})

                if torch.any(torch.isnan(losses[names[0]])):
                    logger.error(f"NaN loss detected: {names[0]} {losses[names[0]]}")
                    logger.error(losses)

            timing["t_loss"] = time.time() - _start_time

        _start_time = time.time()
        optimizer.zero_grad()
        ## make same scale for gradients. Note: it's not ignite built-in func. (c.f. https://wandb.ai/wandb_fc/tips/reports/How-To-Use-GradScaler-in-PyTorch--VmlldzoyMTY5MDA5)
        scaler.scale(overall_loss).backward()

        if config["training"].get("clip_grad", None):
            logger.warning("Clipping gradients")
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["clip_grad"])
            
        scaler.step(optimizer)
        scaler.update()
        timing["t_backward"] = time.time() - _start_time

        return {
            "output": data,
            "loss_dict": loss_metrics,
            "timings_dict": timing,
            "metrics_dict": {},
        }

    trainer = Engine(train_step)
    trainer.logger = logger

    for name, metric in metrics.items():
        metric.attach(trainer, name)

    # TODO: maybe save only the network not the whole wrapper
    # TODO: Make adaptable
    to_save = {
        "trainer": trainer,
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
    }

    common.setup_common_training_handlers(
        trainer=trainer,
        train_sampler=train_sampler,
        to_save=to_save,
        save_every_iters=config["training"]["checkpoint_every"],
        save_handler=DiskSaver(config["output"]["path"], require_empty=False),
        lr_scheduler=lr_scheduler,
        output_names=None,
        with_pbars=False,
        clear_cuda_cache=False,
        log_every_iters=config.get("log_every_iters", 100),
        n_saved=config["training"].get("n_saved", 10),
        stop_on_nan=False,
    )

    # NOTE: don't move to initialization, as to save is also needed here
    model_filters = config["training"].get("model_filter", None)

    logger.info(f"Checking for resume and pretrained checkpoints. {config['training'].get('resume_from', None)}, {config['training'].get('from_pretrained', None)}")

    if config["training"].get("from_pretrained", None):
        from_pretraineds = config["training"]["from_pretrained"]
        if not isinstance(from_pretraineds, list) and not isinstance(from_pretraineds, ListConfig):
            from_pretraineds = [from_pretraineds]
            model_filters = [model_filters]
        
        for ckpt_path, model_filter in zip(from_pretraineds, model_filters):
            ckpt_path = Path(ckpt_path)
            logger.info(f"Pretrained from checkpoint: {str(ckpt_path)}")

            to_save = {"model": to_save["model"]}

            load_checkpoint(ckpt_path, to_save, strict=False, model_filter=model_filter)
    
    if config["training"].get("resume_from", None):
        ckpt_path = Path(config["training"]["resume_from"])
        logger.info(f"Resuming from checkpoint: {str(ckpt_path)}")

        load_checkpoint(ckpt_path, to_save, strict=False, model_filter=None)

    return trainer


def create_validators(
    config,
    model: torch.nn.Module,
    dataloaders: dict[str, DataLoader],
    criterions: list[BaseLoss],
    tb_logger: TensorboardLogger,
    trainer: Engine,
) -> dict[str, tuple[Engine, EventsList]]:
    # TODO: change model object to evaluator object that has a different ray sampler
    with_amp = config["with_amp"]
    device = idist.device()

    def _create_validator(
        model: torch.nn.Module,
        tag: str,
        validation_config,
        with_amp,
        device,
        tb_logger,
        trainer,
        config,
    ) -> tuple[Engine, EventsList]:
        # TODO: make eval functions configurable from config
        metrics = {
            metric_config["type"]: DictMeanMetric(
                metric_config["type"], make_eval_fn(model, metric_config)
            )
            for metric_config in validation_config["metrics"]
        }
        loss_during_validation = validation_config.get("log_loss", True)
        if loss_during_validation:
            metrics_loss = {}
            for criterion in criterions:
                metrics_loss.update(
                    {
                        k: MeanMetric((lambda y: lambda x: x["loss_dict"][y])(k))
                        for k in criterion.get_loss_metric_names()
                    }
                )
            eval_metrics = {**metrics, **metrics_loss}
        else:
            eval_metrics = metrics

        @torch.no_grad()
        def validation_step(engine: Engine, data):
            model.eval()
            if "t__get_item__" in data:
                timing = {"t__get_item__": torch.mean(data["t__get_item__"]).item()}
            else:
                timing = {}

            data = to(data, device)

            with autocast(enabled=with_amp, device_type="cuda"):
                data = model(data, tag=tag)

                overall_loss = torch.tensor(0.0, device=device)
                loss_metrics = {}
                if loss_during_validation:
                    for criterion in criterions:
                        losses = criterion(data)
                        names = criterion.get_loss_metric_names()

                        overall_loss += losses[names[0]]
                        loss_metrics.update({name: loss for name, loss in losses.items()})
                else:
                    loss_metrics = {}

            return {
                "output": data,
                "loss_dict": loss_metrics,
                "timings_dict": timing,
                "metrics_dict": {},
            }

        validator = Engine(validation_step)

        add_time_handlers(validator)

        # ADD METRICS
        for name, metric in eval_metrics.items():
            metric.attach(validator, name)

        # ADD LOGGING HANDLER
        # TODO: split up handlers
        tb_logger.attach(
            validator,
            MetricLoggingHandler(
                tag,
                log_loss=False,
                global_step_transform=global_step_fn(
                    trainer, validation_config["global_step"]
                ),
            ),
            Events.EPOCH_COMPLETED,
        )

        # ADD VISUALIZATION HANDLER
        if validation_config.get("visualize", None) and tag.startswith("visualization"):
            visualize = tb_visualize(
                (model.renderer.net if hasattr(model, "renderer") else model.module.renderer.net),
                dataloaders[tag].dataset,
                validation_config["visualize"],
            )

            def vis_wrapper(*args, **kwargs):
                with autocast(enabled=with_amp, device_type="cuda"):
                    return visualize(*args, **kwargs)

            tb_logger.attach(
                validator,
                VisualizationHandler(
                    tag=tag,
                    visualizer=vis_wrapper,
                    global_step_transform=global_step_fn(
                        trainer, validation_config["global_step"]
                    ),
                ),
                Events.ITERATION_COMPLETED(every=1),
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
            # TODO: check if works
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

        return validator, event_list_from_config(validation_config["events"])

    validators = {}

    for name_val, loader in dataloaders.items():
        val_config = config["validation"][name_val.split("/")[0]]

        if "custom_validator" in val_config:
            path, name = val_config["custom_validator"].rsplit(".", 1)
            create_validator_fn = getattr(__import__(path, fromlist=[name]), name)
        else:
            create_validator_fn = _create_validator

        validators[f"{name_val}"] = create_validator_fn(model, name_val, val_config, with_amp, device, tb_logger, trainer, config)

    return validators
