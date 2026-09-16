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
import matplotlib.pyplot as plt
import numpy as np
import torch
from dlk.mgmt import parameters as config_params
from dlk.mgmt.log import logging_get_logger
from dlk.mode import Mode, get_mode_from_name
from dlk.opt.optimizer import create_optimizer_from_config
from dlk.opt.scheduler import create_learning_rate_scheduler_from_config
from dlk.opt.train import train_epochs
from dlk.opt.utils import checkpoint_load
from nets import create_network
from plot_utils import plot_loss

from data import create_dataloader


def run_train(
    params: dict[str, Any],
    device: torch.device,
    logger: logging.Logger,
) -> None:
    """Run training for a DNN inverse map.

    Builds its own device/logger via ``common.initialize_run`` when either is
    omitted. When both are provided (e.g. by ``run.py``), logging setup is
    skipped so a combined process keeps a single log-file set.

    Args:
        params: Already-loaded configuration dict. Requires
            ``Mode.TRAIN`` in ``params["runconfig"]["mode"]``.
        device: Torch device from ``common.initialize_run``.
        logger: Logger from ``common.initialize_run``.

    Raises:
        ValueError: If ``params["runconfig"]["mode"]`` has no ``Mode.TRAIN`` bit.
    """
    path_file = pathlib.Path(__file__)
    self_dir = path_file.parent
    self_tag = f"{path_file.stem}_{path_file.suffix.lstrip('.')}"

    print(f"<{self_tag}>")

    # get mode
    mode = get_mode_from_name(params["runconfig"]["mode"])
    logger.info(f"Mode: {mode}")
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
    ) = common.load_and_preprocess_data(params, device, logger)

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
    )

    # </data>

    # <network>

    # create network
    net = create_network(params, device, logging_get_logger("create_network"))

    # resume from checkpoint when set
    load_checkpoint = params["runconfig"].get("load_checkpoint")
    if load_checkpoint:
        checkpoint_path = self_dir / load_checkpoint
        epoch = checkpoint_load(checkpoint_path, net, map_location=device)
        logger.info(f"Resume at checkpoint: {checkpoint_path} (epoch {epoch})")

    # </network>

    # <train>

    # create optimizer
    optimizer = create_optimizer_from_config(net, params["optimizer"])

    # create learning rate scheduler
    lr_scheduler = create_learning_rate_scheduler_from_config(
        optimizer, params["optimizer"], params["training"]["epochs"]
    )

    # set loss function
    loss_fn = torch.nn.MSELoss()

    # checkpointing for saving network weights
    checkpoint_dir = self_dir / params["runconfig"]["save_dir"] / "checkpoints"
    checkpoint_epochs = params["runconfig"]["save_checkpoints_epochs"]

    train_dlog: dict[str, Any] | None = None
    time_train: float = np.nan

    if Mode.PROFILE in mode:
        # profile training
        from dlk.opt.profiler import profile_train_batches
        from dlk.opt.train import train_batches

        train_batches_kwargs = dict(
            device=device,
            inputs_transform_fn=train_input_transform_fn,
        )
        log_profile_dir = self_dir / params["runconfig"]["save_dir"] / "profile"

        profile_train_batches(
            train_batches,
            train_batches_kwargs,
            net,
            dataloader,
            optimizer,
            loss_fn,
            log_profile_dir=log_profile_dir,
        )
    else:
        # train network
        print("<train>")
        train_dlog = train_epochs(
            n_epochs=params["training"]["epochs"],
            net=net,
            dataloader=dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            lr_scheduler=lr_scheduler,
            device=device,
            inputs_transform_fn=train_input_transform_fn,
            checkpoint_epochs=checkpoint_epochs,
            checkpoint_dir=checkpoint_dir,
        )
        time_train = train_dlog.get("time_train", np.nan)
        print("</train>")

    # </train>

    # <output>

    show_plots = params["runconfig"].get("show_plots", False)

    # log training runtimes and plot loss (skip for profile-only runs)
    if train_dlog is not None:
        logger.info(f"Runtime [sec]:                         {time_train}")
        n_epoch = params["training"]["epochs"]
        n_steps = params["training"]["epochs"] * (
            params["data_train"]["Ntrain"] // params["data_train"]["train_batch_size"]
        )
        n_samples = params["data_train"]["train_batch_size"]
        logger.info(f"Runtime statistics - #epochs:          {n_epoch}")
        logger.info(f"Runtime statistics - #steps:           {n_steps}")
        logger.info(f"Runtime statistics - #samples (total): {n_steps * n_samples}")
        logger.info(f"Runtime statistics - avg. steps/sec:   {n_steps / time_train}")
        logger.info(
            f"Runtime statistics - avg. samples/sec: {n_steps * n_samples / time_train}"
        )

        # plot loss
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

    print(f"</{self_tag}>")


def main() -> None:
    """Parse CLI args, load params, and run training."""
    # <params>

    parser = argparse.ArgumentParser()
    config_params.add_args_to_parser(
        parser,
        default_params_path="configs/params_dnn.yaml",
        default_mode="train",
    )
    args = parser.parse_args(sys.argv[1:])

    # load parameters from a file
    params = config_params.load(args.params)

    # set/override runconfig parameters from args
    config_params.update_runconfig_params_from_args(params["runconfig"], args)

    # override parameters from JSON and/or TOML inputs
    if args.json_params is not None:
        config_params.update_from_json(params, args.json_params)
    if args.toml_params is not None:
        config_params.update_from_toml(params, args.toml_params)

    # </params>

    # set the device/logger pair
    device, logger = common.initialize_run(
        pathlib.Path(__file__).parent,
        pathlib.Path(__file__).stem,
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
    config_params.save(params, save_dir=params["runconfig"]["save_dir"])

    # train the network
    run_train(params, device, logger)


if __name__ == "__main__":
    main()
