from pathlib import Path
from typing import Any, OrderedDict

import torch
from ignite.handlers import Checkpoint


def load_checkpoint(ckpt_path: Path, to_save: dict[str, Any], strict: bool = False, model_filter=None):
    assert ckpt_path.exists(), f"__Checkpoint '{str(ckpt_path)}' is not found"
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")

    if model_filter is not None:
        model_checkpoint = checkpoint["model"]

        model_checkpoint = OrderedDict({k: v for k, v in model_checkpoint.items() if any(k.startswith(fk) for fk in model_filter)})

        checkpoint["model"] = model_checkpoint

    Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint, strict=strict)
