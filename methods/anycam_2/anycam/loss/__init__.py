from anycam.loss.pose_loss import PoseLoss


def make_loss(config):
    type = config.get("type", "flow_loss")

    if type == "pose_loss":
        return PoseLoss(config)
    else:
        raise ValueError(f"Unknown loss type {type}")