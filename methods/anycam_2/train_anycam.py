import ignite.distributed as idist
import hydra
from omegaconf import DictConfig, OmegaConf
import os

import torch

from anycam.trainer import training as anycam_training


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


@hydra.main(version_base=None, config_path="anycam/configs", config_name="exp_kitti_360_DFT")
def main(config: DictConfig):
    OmegaConf.set_struct(config, False)

    os.environ["NCCL_DEBUG"] = "INFO"

    if config.get("debug", False):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        torch.autograd.set_detect_anomaly(False)

    ## Set up for training with multi-GPUs
    backend = config.get("backend", None)
    nproc_per_node = config.get("nproc_per_node", None)
    with_amp = config.get("with_amp", False)
    spawn_kwargs = {}

    spawn_kwargs["nproc_per_node"] = nproc_per_node
    if backend == "xla-tpu" and with_amp:
        raise RuntimeError("The value of with_amp should be False if backend is xla")

    if config.get("master_port", None):
        spawn_kwargs["master_port"] = config.master_port
    if config.get("init_method", None):
        spawn_kwargs["init_method"] = config.init_method

    training = globals()[
        config["training_type"]
    ]

    with idist.Parallel(
        backend=backend, **spawn_kwargs
    ) as parallel:
        parallel.run(training, config)


if __name__ == "__main__":
    main()
