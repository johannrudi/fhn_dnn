import torch
from dlk.opt.scheduler import create_learning_rate_scheduler

###############################################################################


def create_optimizer(net, opt_params):
    if "Adam".casefold() == opt_params["type"].casefold():
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=opt_params["learning_rate"],
            betas=(opt_params["beta1"], opt_params["beta2"]),
            eps=opt_params["epsilon"],
        )
    elif "AdamW".casefold() == opt_params["type"].casefold():
        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=opt_params["learning_rate"],
            betas=(opt_params["beta1"], opt_params["beta2"]),
            eps=opt_params["epsilon"],
            weight_decay=opt_params["weight_decay"],
        )
    else:
        raise ValueError("Unknown name for optimizer: " + opt_params["type"])
    return optimizer


def create_lr_scheduler(optimizer, opt_params, n_epochs):
    lr_scheduler_params = opt_params.get("learning_rate_scheduler", None)
    if lr_scheduler_params is not None:
        lr_scheduler = create_learning_rate_scheduler(
            optimizer,
            n_epochs,
            opt_params["learning_rate"],
            linear_epochs=lr_scheduler_params["linear_epochs"],
            constant_epochs=lr_scheduler_params["constant_epochs"],
            init_learning_rate=lr_scheduler_params["init_learning_rate"],
            final_learning_rate=lr_scheduler_params["final_learning_rate"],
        )
    else:
        lr_scheduler = None
    return lr_scheduler
