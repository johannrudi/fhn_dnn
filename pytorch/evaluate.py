"""
Evaluate a trained DNN-based inverse map from a checkpoint.

Loads weights via ``dlk.opt.utils.checkpoint_load``. The checkpoint
architecture must match ``params["net"]``.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import timeit
from collections.abc import Callable
from typing import Any

import common
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import torch
from dlk.mgmt import parameters as config_params
from dlk.mgmt.log import logging_get_logger
from dlk.mode import Mode, get_mode_from_name
from dlk.opt.utils import checkpoint_load
from nets import create_network
from plot_utils import plot_data_vs_predict, plot_data_vs_predict_error
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import (
    create_dataloader,
    dictarray_is_not_none,
    postprocess_targets,
)


def run_evaluate(
    params: dict[str, Any],
    device: torch.device | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Run prediction and optional evaluation for a DNN inverse map.

    Loads a checkpoint from ``runconfig.load_checkpoint`` when set; otherwise
    auto-discovers the latest ``*.pt`` under ``runconfig.save_dir/checkpoints``.
    Builds its own device/logger via ``common.initialize_run`` when either is
    omitted.

    Args:
        params: Already-loaded configuration dict. Requires
            ``Mode.PREDICT`` or ``Mode.EVAL`` in ``params["runconfig"]["mode"]``.
        device: Optional pre-built torch device from ``common.initialize_run``.
        logger: Optional pre-built logger from ``common.initialize_run``.

    Raises:
        ValueError: If mode has neither ``PREDICT`` nor ``EVAL``.
        FileNotFoundError: If no checkpoint can be resolved.
    """
    path_file = pathlib.Path(__file__)
    self_dir = path_file.parent
    self_tag = f"{path_file.stem}_{path_file.suffix.lstrip('.')}"

    print(f"<{self_tag}>")

    # <init>

    # set mode
    mode_name = params["runconfig"]["mode"]
    mode = get_mode_from_name(mode_name)
    assert mode is not None
    if not mode.any(Mode.PREDICT | Mode.EVAL):
        raise ValueError(
            f"run_evaluate requires Mode.PREDICT or Mode.EVAL in mode, "
            f"got {mode} (--mode {mode_name})"
        )

    # set device/logger pair
    if device is None or logger is None:
        device, logger = common.initialize_run(self_dir, path_file.stem, params)
    assert device is not None
    assert logger is not None

    logger.info(f"Mode: {mode} (--mode {mode_name})")

    # </init>

    # <data>

    (
        features,
        targets,
        features_noise,
        targets_noise,
        targets_scale,
        targets_noise_scale,
        features_transform_fn,
        train_input_transform_fn,
    ) = common.load_and_preprocess_data(params, device, logger)

    # create dataloaders per split
    eval_dataloader: dict[str, DataLoader] = dict()
    for key in features.keys():
        eval_dataloader[key] = create_dataloader(
            params,
            logging_get_logger("create_dataloader"),
            Mode.EVAL,
            features=features[key],
            targets=targets[key],
            features_noise=features_noise[key],
            targets_noise=targets_noise[key],
            features_transform_fn=features_transform_fn,
        )

    # </data>

    # <network>

    # create network
    net = create_network(params, device, logging_get_logger("create_network"))

    # resolve checkpoint path
    load_checkpoint = params["runconfig"].get("load_checkpoint")
    if load_checkpoint:
        checkpoint_path = self_dir / load_checkpoint
    else:
        checkpoint_path = common.find_latest_checkpoint(
            self_dir / params["runconfig"]["save_dir"]
        )

    epoch = checkpoint_load(checkpoint_path, net, map_location=device)
    logger.info(f"Evaluate at checkpoint: {checkpoint_path} (epoch {epoch})")

    # </network>

    # <predict>

    print("<predict>")

    # compute predictions
    time_eval = timeit.default_timer()
    eval_targets_pred, eval_targets_data = predict(
        net,
        eval_dataloader,
        device,
        input_transform_fn=train_input_transform_fn,
    )
    time_eval = timeit.default_timer() - time_eval

    # postprocess evaluation data
    if dictarray_is_not_none(targets) and dictarray_is_not_none(targets_noise):
        assert targets_scale is not None
        assert targets_noise_scale is not None
        eval_targets_scale = {}
        for key in targets_scale.keys():
            eval_targets_scale[key] = np.concatenate(
                (targets_scale[key], targets_noise_scale[key]), axis=1
            )
        postprocess_targets(eval_targets_data, eval_targets_scale)
        postprocess_targets(eval_targets_pred, eval_targets_scale)
    elif dictarray_is_not_none(targets):
        postprocess_targets(eval_targets_data, targets_scale)
        postprocess_targets(eval_targets_pred, targets_scale)
    elif dictarray_is_not_none(targets_noise):
        postprocess_targets(eval_targets_data, targets_noise_scale)
        postprocess_targets(eval_targets_pred, targets_noise_scale)
    else:
        raise NotImplementedError()

    print("</predict>")

    # </predict>

    if Mode.EVAL in mode:
        print("<evaluate>")

        # compute evaluation metrics
        eval_mse, eval_mae, eval_r2 = eval_data_vs_pred(
            eval_targets_data, eval_targets_pred
        )
        for key in eval_targets_data.keys():
            logger.info(
                f"MSE ({key}):      " + str(eval_mse[key + "_i"]) + f" {eval_mse[key]}"
            )
            logger.info(
                f"MAE ({key}):      " + str(eval_mae[key + "_i"]) + f" {eval_mae[key]}"
            )
            logger.info(
                f"R2 score ({key}): " + str(eval_r2[key + "_i"]) + f" {eval_r2[key]}"
            )

        print("</evaluate>")

    # </evaluate>

    # <output>

    # log eval runtimes
    logger.info(f"Runtime [sec]:                         {time_eval}")
    n_samples = (
        params["data_evaluate"]["Ntest"] // params["data_evaluate"]["eval_batch_size"]
    ) * params["data_evaluate"]["eval_batch_size"]
    logger.info(f"Runtime statistics - #samples:         {n_samples}")
    logger.info(f"Runtime statistics - avg. samples/sec: {n_samples / time_eval}")

    # plot predictions
    for key in eval_targets_data.keys():
        # skip if no samples exist
        n_key = "N" + key
        n_samples_for_key = params["data_train"].get(
            n_key, params["data_evaluate"].get(n_key)
        )
        if n_samples_for_key is None or n_samples_for_key <= 0:
            continue
        # set up plotting
        assert eval_targets_data[key].shape[1] == eval_targets_pred[key].shape[1]
        ntrg = eval_targets_data[key].shape[1]
        plot_targets_data = [eval_targets_data[key][:, i] for i in range(ntrg)]
        plot_targets_pred = [eval_targets_pred[key][:, i] for i in range(ntrg)]
        plot_name = [f"param_{i}" for i in range(ntrg)]
        # plot true values vs. predictions
        path = self_dir / params["runconfig"]["save_dir"] / f"data_vs_predict_{key}"
        plot_data_vs_predict(
            plot_targets_data,
            plot_targets_pred,
            path,
            plot_name=plot_name,
            x_label=ntrg * [f"{key} value"],
            y_label=ntrg * ["predicted value"],
        )
        # plot prediction errors
        path = self_dir / params["runconfig"]["save_dir"] / f"predict_error_{key}"
        plot_data_vs_predict_error(
            plot_targets_data,
            plot_targets_pred,
            path,
            plot_name=plot_name,
            x_label=ntrg * [f"{key} value"],
            y_label=ntrg * ["prediction error"],
        )
    if not params["runconfig"]["show_plots"]:
        plt.close()

    # show plots
    if params["runconfig"]["show_plots"]:
        plt.show()

    # <output>

    print(f"</{self_tag}>")


def predict(
    net: torch.nn.Module,
    eval_dataloader: dict[str, DataLoader],
    device: torch.device,
    input_transform_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Run the network over each eval dataloader split.

    Args:
        net: Network in eval mode after this call.
        eval_dataloader: Mapping from split name to dataloader.
        device: Device for inputs and the forward pass.
        input_transform_fn: Optional transform applied to each batch input.

    Returns:
        ``(pred, data)`` dicts keyed by split, each value an ``ndarray``.
    """
    net.eval()
    # get network predictions
    data: dict[str, np.ndarray] = dict()
    pred: dict[str, np.ndarray] = dict()
    with torch.no_grad():
        for key in eval_dataloader.keys():
            d_list: list[np.ndarray] = list()
            p_list: list[np.ndarray] = list()
            for x, yd in tqdm(eval_dataloader[key], desc=key):
                x = x.to(device)
                if input_transform_fn is not None:
                    x = input_transform_fn(x)
                yp = net(x)
                d_list.append(yd.cpu().numpy())
                p_list.append(yp.cpu().numpy())
            data[key] = np.concatenate(d_list, axis=0)
            pred[key] = np.concatenate(p_list, axis=0)
    # return predictions and (true) data
    return pred, data


def eval_data_vs_pred(
    data: dict[str, np.ndarray],
    pred: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compute per-target and overall MSE, MAE, and R2 for each split.

    Args:
        data: Mapping from split name to ground-truth ``ndarray``.
        pred: Mapping from split name to predicted ``ndarray``.

    Returns:
        ``(eval_mse, eval_mae, eval_r2)`` dicts with per-index ``key_i``
        lists and overall ``key`` scalars.
    """
    eval_mse: dict[str, Any] = dict()
    eval_mae: dict[str, Any] = dict()
    eval_r2: dict[str, Any] = dict()
    for key in data.keys():
        data_ = data[key]
        pred_ = pred[key]
        eval_mse[key + "_i"] = [
            metrics.mean_squared_error(data_[:, i], pred_[:, i])
            for i in range(data_.shape[1])
        ]
        eval_mse[key] = metrics.mean_squared_error(data_, pred_)
        eval_mae[key + "_i"] = [
            metrics.mean_absolute_error(data_[:, i], pred_[:, i])
            for i in range(data_.shape[1])
        ]
        eval_mae[key] = metrics.mean_absolute_error(data_, pred_)
        eval_r2[key + "_i"] = [
            metrics.r2_score(data_[:, i], pred_[:, i]) for i in range(data_.shape[1])
        ]
        eval_r2[key] = metrics.r2_score(data_, pred_)
    return eval_mse, eval_mae, eval_r2


def main() -> None:
    """Parse CLI args, load params, and run evaluation."""
    # <params>

    parser = argparse.ArgumentParser()
    config_params.add_args_to_parser(
        parser,
        default_params_path="configs/params_dnn.yaml",
        default_mode="eval",
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

    # reject modes that belong to train.py / run.py
    mode = get_mode_from_name(params["runconfig"]["mode"])
    allowed_modes = {Mode.PREDICT, Mode.EVAL}
    if mode not in allowed_modes:
        raise ValueError(
            f"evaluate.py only allows modes 'predict' and 'eval', "
            f"got {mode} (--mode {params['runconfig']['mode']}). "
            "Use train.py for train/train_profile, or run.py for combined modes."
        )

    # save parameters for reproducibility
    config_params.save(params, save_dir=params["runconfig"]["save_dir"])

    # evaluate the network
    run_evaluate(params)


if __name__ == "__main__":
    main()
