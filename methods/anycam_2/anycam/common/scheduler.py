from torch import optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts


class FixLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, last_epoch=-1, verbose=False):
        super(FixLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return [base_lr for base_lr in self.base_lrs]

    def _get_closed_form_lr(self):
        return [base_lr for base_lr in self.base_lrs]


def make_scheduler(config, optim):
    type = config.get("type", "fix")
    if type == "fix":
        scheduler = FixLR(optim)
        return scheduler
    elif type == "step":
        scheduler = StepLR(
            optim,
            config["step_size"],
            config["gamma"]
        )
        return scheduler
    elif type == "cosine-warmrestarts":
        scheduler = CosineAnnealingWarmRestarts(
            optim,
            T_0=config["T_0"],
            T_mult=config["T_mult"],
            eta_min=config["eta_min"]
        )
        return scheduler
    else:
        raise NotImplementedError(f"Unknown learning rate scheduler type: {type}")
