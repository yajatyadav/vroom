from datetime import datetime
from pathlib import Path

import ignite.distributed as idist
import torch
from ignite.contrib.engines import common
from ignite.engine import Engine, Events
from ignite.utils import manual_seed, setup_logger
from torch.cuda.amp import autocast

from anycam.common.logging import log_basic_info
from anycam.common.array_operations import to
from anycam.common.metrics import DictMeanMetric
from anycam.training.handlers import MetricLoggingHandler
from anycam.training.eval_handler import make_eval_fn


def base_evaluation(
    local_rank,
    config,
    get_dataflow,
    initialize,
):
    rank = idist.get_rank()
    manual_seed(config["seed"] + rank)
    device = idist.device()

    model_name = config["name"]
    logger = setup_logger(
        name=model_name, format="%(levelname)s: %(message)s"
    )  ## default

    log_basic_info(logger, config)

    # Setup dataflow, model, optimizer, criterion
    test_loader = get_dataflow(config)  ## default

    if not isinstance(test_loader, dict):
        test_loader = {"default": test_loader}

    for name, loader in test_loader.items():
        if hasattr(loader, "dataset"):
            logger.info(f"Dataset length - {name}: {len(loader.dataset)}")


    # ===================================================== MODEL =====================================================
    model = initialize(config)

    cp_path = config.get("checkpoint", None)

    if cp_path is not None:
        if not cp_path.endswith(".pt"):
            cp_path = Path(cp_path)
            cp_path = next(cp_path.glob("training*.pt"))
        checkpoint = torch.load(cp_path, map_location=device)
        logger.info(f"Loading checkpoint from path: {cp_path}")
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        logger.warning("Careful, no model is loaded")
    model.to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Let's now setup evaluator engine to perform model's validation and compute metrics
    evaluator = create_evaluator(model, config=config, logger=logger)

    for name, loader in test_loader.items():
        try:
            state = evaluator.run(loader, max_epochs=1)
            log_metrics(logger, state.times["COMPLETED"], name, state.metrics)
            logger.info(f"Checkpoint: {str(cp_path)}")
        except Exception as e:
            logger.exception("")
            raise e


def log_metrics(logger, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(
        f"\nEvaluation time (seconds): {elapsed:.2f} - {tag} metrics:\n {metrics_output}"
    )


# def create_evaluator(model, metrics, config, tag="val"):
def create_evaluator(model, config, logger=None, tag="val"):
    with_amp = config["with_amp"]
    device = idist.device()

    metrics = {
        eval_conf["type"]: DictMeanMetric(
            eval_conf["type"], make_eval_fn(model, eval_conf)
        )
        for eval_conf in config["evaluations"]
    }

    @torch.no_grad()
    def evaluate_step(engine: Engine, data):
        model.eval()
        if "t__get_item__" in data:
            timing = {"t__get_item__": torch.mean(data["t__get_item__"]).item()}
        else:
            timing = {}

        data = to(data, device)

        with torch.autocast(enabled=with_amp, device_type="cuda"):
            data = model(data, tag="validation")  ## ! This is where the occupancy prediction is made.

        loss_metrics = {}

        return {
            "output": data,
            "loss_dict": loss_metrics,
            "timings_dict": timing,
            "metrics_dict": {},
        }

    evaluator = Engine(evaluate_step)
    evaluator.logger = logger  ##

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # TODO: figure out how to log in regular intervals

    if idist.get_rank() == 0 and (not config.get("with_clearml", False)):
        common.ProgressBar(desc=f"Evaluation ({tag})", persist=False).attach(evaluator)

    return evaluator
