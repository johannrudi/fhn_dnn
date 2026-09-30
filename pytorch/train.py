"""
Train a DNN-based inverse map and write checkpoints.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
from typing import Any

import common
import dlk.mgmt.parameters as parameters
import dlk.opt.distributed as distributed
import matplotlib.pyplot as plt
import torch
from dlk.mgmt.log import logging_get_logger
from dlk.mode import Mode, get_mode_from_name
from dlk.opt.compile import compile_net_from_params
from dlk.opt.optimizer import create_optimizer_from_params
from dlk.opt.scheduler import create_learning_rate_scheduler_from_params
from dlk.opt.train import train_epochs
from dlk.opt.utils import checkpoint_load
from nets import create_network
from plot_utils import plot_loss

from data import create_dataloader

AUTOCAST_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def run_train(
    params: dict[str, Any],
    ctx: distributed.DistributedContext,
    logger: logging.Logger,
) -> None:
    """Run training for a DNN inverse map.

    Args:
        params: Already-loaded configuration dict. Requires
            ``Mode.TRAIN`` in ``params["runconfig"]["mode"]``.
        ctx: Distributed context from ``distributed.session``.
        logger: Logger from ``common.initialize_run``.

    Raises:
        ValueError: If ``params["runconfig"]["mode"]`` has no ``Mode.TRAIN`` bit.
    """
    path_file = pathlib.Path(__file__)
    self_dir = path_file.parent
    self_tag = f"{path_file.stem}_{path_file.suffix.lstrip('.')}"

    if distributed.is_main_process():
        print(f"<{self_tag}>")

    # get mode
    mode = get_mode_from_name(params["runconfig"]["mode"])
    logger.info(mode)
    assert mode is not None
    if Mode.TRAIN not in mode:
        raise ValueError(f"run_train requires Mode.TRAIN in mode, got {mode}")

    # <data>

    (
        features,
        targets,
        features_noise,
        targets_noise,
        _,
        _,
        features_transform_fn,
        train_input_transform_fn,
    ) = common.load_and_preprocess_data(params, ctx.device, logger)

    # create training dataloader
    dataloader = create_dataloader(
        params,
        logging_get_logger("create_dataloader"),
        mode,
        features=features["train"],
        targets=targets["train"],
        features_noise=features_noise["train"],
        targets_noise=targets_noise["train"],
        features_transform_fn=features_transform_fn,
        base_seed=params["runconfig"].get("random_seed") or 0,
        with_distributed=ctx.is_distributed,
    )

    # </data>

    # <network>

    # create network
    net = create_network(params, ctx.device)

    # resume from checkpoint when set (before wrapping for DDP)
    load_checkpoint = params["runconfig"].get("load_checkpoint")
    if load_checkpoint:
        checkpoint_path = self_dir / load_checkpoint
        epoch = checkpoint_load(checkpoint_path, net, map_location=ctx.device)
        logger.info(f"Resume at checkpoint: {checkpoint_path} (epoch {epoch})")

    # wrap for DDP (no-op unless distributed)
    net = distributed.wrap_net(net, ctx.device)

    # compile the network after wrapping for DDP (no-op unless enabled in the config)
    net = compile_net_from_params(net, params["training"].get("compile"))

    # set mixed precision
    autocast_dtype = params["training"].get("autocast_dtype")
    if autocast_dtype:
        autocast_dtype = AUTOCAST_DTYPES[autocast_dtype]
        logger.info(f"Enable autocast with dtype={autocast_dtype}")

    # </network>

    # <train>

    # create optimizer
    optimizer = create_optimizer_from_params(net, params["optimizer"])

    # create learning rate scheduler
    lr_scheduler = create_learning_rate_scheduler_from_params(
        optimizer, params["optimizer"], params["training"]["epochs"]
    )

    # set loss function
    loss_fn = torch.nn.MSELoss()

    # checkpointing for saving network weights
    checkpoint_dir = self_dir / params["runconfig"]["save_dir"] / "checkpoints"
    checkpoint_epochs = params["runconfig"]["save_checkpoints_epochs"]

    train_dlog: dict[str, Any] | None = None

    if Mode.PROFILE in mode:
        from dlk.opt.profiler import profile_train_batches
        from dlk.opt.train import train_batches

        train_batches_args = (
            net,
            dataloader,
            optimizer,
            loss_fn,
        )
        train_batches_kwargs: dict[str, Any] = dict(
            device=ctx.device,
            inputs_transform_fn=train_input_transform_fn,
            autocast_dtype=autocast_dtype,
        )
        trace_dir = self_dir / params["runconfig"]["save_dir"] / "profile"

        # profile training
        if distributed.is_main_process():
            print("<train_profile>")
        profile_train_batches(
            train_batches,
            train_batches_args,
            train_batches_kwargs,
            skip_first=params["training"].get("profile_skip_first", 0),
            trace_dir=trace_dir,
        )
        if distributed.is_main_process():
            print("</train_profile>")
    else:
        # train network
        if distributed.is_main_process():
            print("<train>")
        train_dlog = train_epochs(
            n_epochs=params["training"]["epochs"],
            net=net,
            dataloader=dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            lr_scheduler=lr_scheduler,
            device=ctx.device,
            inputs_transform_fn=train_input_transform_fn,
            checkpoint_epochs=checkpoint_epochs,
            checkpoint_dir=checkpoint_dir,
            autocast_dtype=autocast_dtype,
        )
        if distributed.is_main_process():
            print("</train>")

    # </train>

    # <output>

    if distributed.is_main_process():
        show_plots = params["runconfig"].get("show_plots", False)

        # plot loss (skip for profile-only runs)
        if train_dlog is not None:
            path = self_dir / params["runconfig"]["save_dir"] / "loss"
            plot_loss(
                loss=train_dlog["loss_mean"],
                path=path,
                plot_name="Training loss",
                n_epochs=params["training"]["epochs"],
                loss_std=train_dlog["loss_std"],
                x_offset=1,
                y_scale="log",
                close_plot=not show_plots,
            )

        # show plots
        if show_plots:
            plt.show()
        else:
            plt.close("all")

    # </output>

    if distributed.is_main_process():
        print(f"</{self_tag}>")


def main() -> None:
    """Parse CLI args, load params, and run training."""
    # initialize distributed parallelism
    with distributed.session() as ctx:

        # <params>

        parser = argparse.ArgumentParser()
        parameters.add_args_to_parser(
            parser,
            default_params_path="configs/params_dnn.yaml",
            default_mode="train",
        )
        args = parser.parse_args(sys.argv[1:])

        # load parameters from a file
        params = parameters.load(args.params)

        # set/override runconfig parameters from args
        parameters.override_runconfig_from_args(params["runconfig"], args)

        # override parameters from JSON and/or TOML inputs
        parameters.override_params_from_args(params, args)

        # </params>

        # initialize
        logger = common.initialize_run(
            pathlib.Path(__file__).parent,
            pathlib.Path(__file__).stem,
            ctx,
            params,
        )

        # reject modes that belong to evaluate.py / run.py
        mode = get_mode_from_name(params["runconfig"]["mode"])
        allowed_modes = {Mode.TRAIN, Mode.TRAIN | Mode.PROFILE}
        if mode not in allowed_modes:
            raise ValueError(
                f"train.py only allows modes 'train' and 'train_profile', "
                f"got {mode} (--mode {params['runconfig']['mode']}). "
                "Use evaluate.py for predict/eval, or run.py for combined modes."
            )

        # save parameters for reproducibility
        if distributed.is_main_process():
            parameters.save(params, save_dir=params["runconfig"]["save_dir"])

        # train the network
        run_train(params, ctx, logger)


if __name__ == "__main__":
    main()
