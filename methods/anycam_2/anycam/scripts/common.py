from pathlib import Path

from omegaconf import OmegaConf
import torch

from anycam.trainer import AnyCamWrapper


def get_checkpoint_path(model_path: Path):
    if not model_path.is_dir():
        return model_path

    prefix = "training_checkpoint_"
    ckpts = Path(model_path).glob(f"{prefix}*.pt")

    training_steps = [int(ckpt.stem.split(prefix)[1]) for ckpt in ckpts]

    ckpt_path = f"{prefix}{max(training_steps)}.pt"
    ckpt_path = Path(model_path) / ckpt_path

    return ckpt_path


def load_model(config: OmegaConf, checkpoint_path: Path, config_overwrite: dict = None):
    model_conf = config["model"]
    model_conf["train_directions"] = "forward"

    if config_overwrite is not None:
        model_conf = OmegaConf.merge(model_conf, OmegaConf.create(config_overwrite))
    
    model = AnyCamWrapper(model_conf)

    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model_cp = cp["model"]

    # model_cp = {
    #     k: v for k, v in model_cp.items() if not (k.startswith("depth_predictor") or k.startswith("image_processor"))# or ("pose_head" in k))
    # }

    model.load_state_dict(model_cp, strict=False)

    return model


