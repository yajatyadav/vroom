import logging
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

import ignite.distributed as idist

from anycam.common.io.configs import load_model_config
from anycam.training.base_evaluator import base_evaluation

from anycam.models import make_depth_predictor, make_pose_predictor
from anycam.trainer import AnyCamWrapper
from anycam.datasets import make_datasets

IDX = 0

logger = logging.getLogger("evaluation")


def evaluation(local_rank, config):
    return base_evaluation(local_rank, config, get_dataflow, initialize)


def get_dataflow(config):

    test_loaders = {}

    dataset_list = config["dataset"]
    dataset_cfgs = [
        config["dataset_configs"][dataset_name] for dataset_name in dataset_list
    ]
    
    for dataset_cfg in dataset_cfgs:
        _, test_dataset = make_datasets(dataset_cfg)

        test_loader = DataLoader(
            Subset(test_dataset, list(range(0, 1024))),
            # test_dataset,
            batch_size=config.get("batch_size", 1),
            num_workers=config["num_workers"],
            shuffle=False,
            drop_last=False,
        )
        
        test_loaders[dataset_cfg["type"]] = test_loader

    return test_loaders


def initialize(config: dict):
    checkpoint = Path(config["checkpoint"])
    logger.info(f"Loading model config from {checkpoint.parent}")
    load_model_config(checkpoint.parent, config)

    model_conf = config["model"]

    depth_predictor = make_depth_predictor(model_conf["depth_predictor"])
    uncert_pose_predictor = make_pose_predictor(model_conf["uncert_pose_predictor"])

    model = AnyCamWrapper(model_conf, depth_predictor, uncert_pose_predictor)

    model = idist.auto_model(model)

    return model
