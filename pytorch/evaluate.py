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
from typing import Any, Sized, cast

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
from plot_utils import (
    plot_data_vs_predict,
    plot_data_vs_predict_error,
    plot_metrics_vs_checkpoint,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import (
    create_dataloader,
    dictarray_is_not_none,
)


def run_evaluate(
    params: dict[str, Any],
    device: torch.device,
    logger: logging.Logger,
) -> None:
    """Run prediction and optional evaluation for a DNN inverse map.

    Two independent passes share one network and the same low-level helpers:

    - ``train``/``validate``: every checkpoint under ``runconfig.save_dir``
      via ``common.find_all_checkpoints``; plots go under
      ``save_dir/checkpoints_eval/<stem>/``; metrics-vs-checkpoint plots
      are written when ``Mode.EVAL`` is set.
    - ``test``: one checkpoint (``runconfig.load_checkpoint`` when set,
      else ``common.find_latest_checkpoint``); plots stay at flat
      ``save_dir``.

    Builds its own device/logger via ``common.initialize_run`` when either is
    omitted.

    Args:
        params: Already-loaded configuration dict. Requires
            ``Mode.PREDICT`` or ``Mode.EVAL`` in ``params["runconfig"]["mode"]``.
        device: Torch device from ``common.initialize_run``.
        logger: Logger from ``common.initialize_run``.

    Raises:
        ValueError: If mode has neither ``PREDICT`` nor ``EVAL``.
        FileNotFoundError: If no checkpoint can be resolved.
    """
    path_file = pathlib.Path(__file__)
    self_dir = path_file.parent
    self_tag = f"{path_file.stem}_{path_file.suffix.lstrip('.')}"

    print(f"<{self_tag}>")

    # get mode
    mode = get_mode_from_name(params["runconfig"]["mode"])
    logger.info(f"Mode: {mode}")
    assert mode is not None
    if not mode.any(Mode.PREDICT | Mode.EVAL):
        raise ValueError(
            f"run_evaluate requires Mode.PREDICT or Mode.EVAL in mode, got {mode}"
        )

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

    # create train/validate and test dataloaders separately
    train_validate_dataloader: dict[str, DataLoader] = {
        key: create_dataloader(
            params,
            logging_get_logger("create_dataloader"),
            Mode.EVAL,
            features=features[key],
            targets=targets[key],
            features_noise=features_noise[key],
            targets_noise=targets_noise[key],
            features_transform_fn=features_transform_fn,
        )
        for key in ("train", "validate")
    }
    test_dataloader: dict[str, DataLoader] = {
        "test": create_dataloader(
            params,
            logging_get_logger("create_dataloader"),
            Mode.EVAL,
            features=features["test"],
            targets=targets["test"],
            features_noise=features_noise["test"],
            targets_noise=targets_noise["test"],
            features_transform_fn=features_transform_fn,
        )
    }

    # </data>

    # <network>

    # create network once; weights are reloaded per checkpoint below
    net = create_network(params, device, logging_get_logger("create_network"))

    # </network>

    save_dir = self_dir / params["runconfig"]["save_dir"]
    show_plots = params["runconfig"].get("show_plots", False)

    # <evaluate_train_validate>

    checkpoints = common.find_all_checkpoints(save_dir)
    metrics_by_epoch: dict[str, dict[str, list]] = {
        "train": {"epoch": [], "mse": [], "mae": [], "r2": []},
        "validate": {"epoch": [], "mse": [], "mae": [], "r2": []},
    }
    n_samples_train_validate = sum(
        len(cast(Sized, dl.dataset)) for dl in train_validate_dataloader.values()
    )

    for checkpoint_path in checkpoints:
        epoch = checkpoint_load(checkpoint_path, net, map_location=device)

        print(f"<evaluate_train_validate checkpoint={checkpoint_path}>")
        logger.info(
            f"Evaluate checkpoint {checkpoint_path} (epoch {epoch}) for train/validate"
        )
        tag = f" [{checkpoint_path.stem}]"

        eval_targets_pred, eval_targets_data, time_eval = _predict_and_postprocess(
            net,
            train_validate_dataloader,
            device,
            targets,
            targets_noise,
            targets_scale,
            targets_noise_scale,
            train_input_transform_fn,
        )
        _log_runtime(logger, time_eval, n_samples_train_validate, tag=tag)
        split_metrics = _log_eval_metrics(
            eval_targets_data, eval_targets_pred, mode, logger, tag=tag
        )
        if split_metrics is not None:
            for key in ("train", "validate"):
                metrics_by_epoch[key]["epoch"].append(epoch)
                metrics_by_epoch[key]["mse"].append(split_metrics[key]["mse"])
                metrics_by_epoch[key]["mae"].append(split_metrics[key]["mae"])
                metrics_by_epoch[key]["r2"].append(split_metrics[key]["r2"])
        _plot_predictions(
            eval_targets_data,
            eval_targets_pred,
            params,
            save_dir / "checkpoints_eval" / checkpoint_path.stem,
            close_plot=not show_plots,
        )

        print(f"</evaluate_train_validate>")

    # aggregate MSE/MAE/R2 vs checkpoint epoch (train + validate on one figure)
    # NOTE: Do not plot the first entry, which is epoch=0 with high errors.
    if Mode.EVAL in mode and checkpoints:
        plot_metrics_vs_checkpoint(
            metrics_by_epoch["train"]["epoch"][1:],
            metrics_by_epoch["train"]["mse"][1:],
            metrics_by_epoch["train"]["mae"][1:],
            metrics_by_epoch["train"]["r2"][1:],
            metrics_by_epoch["validate"]["mse"][1:],
            metrics_by_epoch["validate"]["mae"][1:],
            metrics_by_epoch["validate"]["r2"][1:],
            save_dir / "metrics_vs_checkpoint",
            close_plot=not show_plots,
        )

    # </evaluate_train_validate>

    # <evaluate_test>

    # resolve checkpoint path (unchanged from single-checkpoint flow)
    load_checkpoint = params["runconfig"].get("load_checkpoint")
    if load_checkpoint:
        test_checkpoint_path = self_dir / load_checkpoint
        epoch = checkpoint_load(test_checkpoint_path, net, map_location=device)
    else:
        test_checkpoint_path = common.find_latest_checkpoint(save_dir)
        assert test_checkpoint_path == checkpoint_path

    print(f"<evaluate_test checkpoint={test_checkpoint_path}>")
    logger.info(
        f"Evaluate at checkpoint: {test_checkpoint_path} (epoch {epoch}) for test"
    )

    eval_targets_pred, eval_targets_data, time_eval = _predict_and_postprocess(
        net,
        test_dataloader,
        device,
        targets,
        targets_noise,
        targets_scale,
        targets_noise_scale,
        train_input_transform_fn,
    )
    _log_runtime(
        logger, time_eval, len(cast(Sized, test_dataloader["test"].dataset)), tag=""
    )
    _log_eval_metrics(eval_targets_data, eval_targets_pred, mode, logger, tag="")
    _plot_predictions(
        eval_targets_data,
        eval_targets_pred,
        params,
        save_dir,
        close_plot=not show_plots,
    )

    print(f"</evaluate_test checkpoint>")

    # </evaluate_test>

    # <output>

    # show plots
    if show_plots:
        plt.show()
    else:
        plt.close("all")

    # </output>

    print(f"</{self_tag}>")


def _undo_targets_scale(
    eval_dict: dict[str, np.ndarray],
    scale: dict[str, np.ndarray],
) -> None:
    """Apply inverse target scale to whatever splits are present in ``eval_dict``.

    Mirrors ``data._apply_scale_inverse`` but does not require the full
    train/validate/test key set that ``postprocess_targets`` demands via
    ``dictarray_is_none``.

    Args:
        eval_dict: Split name to array mapping; updated in place.
        scale: Dict with ``shift`` and ``mult`` arrays broadcast over samples.
    """
    for key in eval_dict.keys():
        eval_dict[key] = eval_dict[key] * scale["mult"] + scale["shift"]


def _predict_and_postprocess(
    net: torch.nn.Module,
    eval_dataloader: dict[str, DataLoader],
    device: torch.device,
    targets: Any,
    targets_noise: Any,
    targets_scale: Any,
    targets_noise_scale: Any,
    train_input_transform_fn: Callable[[torch.Tensor], torch.Tensor] | None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float]:
    """Predict on ``eval_dataloader`` and undo target scaling.

    Args:
        net: Network whose weights are already loaded for this pass.
        eval_dataloader: Split name to dataloader mapping for this pass.
        device: Device for the forward pass.
        targets: Full targets dict (used only for scale-selection branching).
        targets_noise: Full noise-targets dict (scale-selection branching).
        targets_scale: Shift/mult scale for ODE targets, or None.
        targets_noise_scale: Shift/mult scale for noise targets, or None.
        train_input_transform_fn: Optional input transform passed to ``predict``.

    Returns:
        ``(eval_targets_pred, eval_targets_data, time_eval)`` where
        ``time_eval`` is the wall-clock seconds spent in ``predict``.
    """
    # compute predictions
    time_eval = timeit.default_timer()
    eval_targets_pred, eval_targets_data = predict(
        net,
        eval_dataloader,
        device,
        input_transform_fn=train_input_transform_fn,
    )
    time_eval = timeit.default_timer() - time_eval

    # postprocess evaluation data (scale inverse on present splits only;
    # postprocess_targets requires train/validate/test keys via dictarray_is_none)
    if dictarray_is_not_none(targets) and dictarray_is_not_none(targets_noise):
        assert targets_scale is not None
        assert targets_noise_scale is not None
        eval_targets_scale = {}
        for key in targets_scale.keys():
            eval_targets_scale[key] = np.concatenate(
                (targets_scale[key], targets_noise_scale[key]), axis=1
            )
        _undo_targets_scale(eval_targets_data, eval_targets_scale)
        _undo_targets_scale(eval_targets_pred, eval_targets_scale)
    elif dictarray_is_not_none(targets):
        _undo_targets_scale(eval_targets_data, targets_scale)
        _undo_targets_scale(eval_targets_pred, targets_scale)
    elif dictarray_is_not_none(targets_noise):
        _undo_targets_scale(eval_targets_data, targets_noise_scale)
        _undo_targets_scale(eval_targets_pred, targets_noise_scale)
    else:
        raise NotImplementedError()

    return eval_targets_pred, eval_targets_data, time_eval


def _log_eval_metrics(
    eval_targets_data: dict[str, np.ndarray],
    eval_targets_pred: dict[str, np.ndarray],
    mode: Mode,
    logger: logging.Logger,
    tag: str = "",
) -> dict[str, dict[str, float]] | None:
    """Log MSE/MAE/R2 per split when ``Mode.EVAL`` is set.

    Args:
        eval_targets_data: Ground-truth arrays keyed by split.
        eval_targets_pred: Predicted arrays keyed by split.
        mode: Current run mode; metrics run only when ``Mode.EVAL in mode``.
        logger: Logger for the metric lines.
        tag: Optional suffix after the split name (e.g. ``" [net_e0040]"``).
            Empty string keeps the untagged ``test``-path format.

    Returns:
        ``None`` when ``Mode.EVAL`` is not in ``mode``; otherwise
        ``{split: {"mse", "mae", "r2"}}`` overall scalars per split key.
    """
    if Mode.EVAL not in mode:
        return None

    # compute evaluation metrics
    eval_mse, eval_mae, eval_r2 = eval_data_vs_pred(
        eval_targets_data, eval_targets_pred
    )
    split_metrics: dict[str, dict[str, float]] = {}
    for key in eval_targets_data.keys():
        logger.info(
            f"MSE ({key}){tag}:      " + str(eval_mse[key + "_i"]) + f" {eval_mse[key]}"
        )
        logger.info(
            f"MAE ({key}){tag}:      " + str(eval_mae[key + "_i"]) + f" {eval_mae[key]}"
        )
        logger.info(
            f"R2 score ({key}){tag}: " + str(eval_r2[key + "_i"]) + f" {eval_r2[key]}"
        )
        split_metrics[key] = {
            "mse": eval_mse[key],
            "mae": eval_mae[key],
            "r2": eval_r2[key],
        }

    return split_metrics


def _log_runtime(
    logger: logging.Logger,
    time_eval: float,
    n_samples: int,
    tag: str = "",
) -> None:
    """Log wall-clock runtime and samples/sec for one predict pass.

    Args:
        logger: Logger for the runtime lines.
        time_eval: Seconds spent in ``predict`` for this pass.
        n_samples: True sample count (``len(dataset)`` summed over splits).
        tag: Optional suffix after ``Runtime`` (e.g. ``" [net_e0040]"``).
            Empty string keeps the untagged ``test``-path format.
    """
    logger.info(
        f"time{tag} {time_eval} s, #samples {n_samples}, samples/sec mean {n_samples / time_eval}"
    )


def _plot_predictions(
    eval_targets_data: dict[str, np.ndarray],
    eval_targets_pred: dict[str, np.ndarray],
    params: dict[str, Any],
    output_dir: pathlib.Path,
    close_plot: bool = True,
) -> None:
    """Write data-vs-predict and error plots for each split in ``output_dir``.

    Args:
        eval_targets_data: Ground-truth arrays keyed by split.
        eval_targets_pred: Predicted arrays keyed by split.
        params: Full config; used to skip splits with ``N* <= 0``.
        output_dir: Directory that receives ``data_vs_predict_<key>`` and
            ``predict_error_<key>`` plot stems. Created if missing.
        close_plot: Forwarded to plot helpers; True closes each figure after
            save (batch runs), False keeps figures open for ``plt.show()``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

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
        path = output_dir / f"data_vs_predict_{key}"
        plot_data_vs_predict(
            plot_targets_data,
            plot_targets_pred,
            path,
            plot_name=plot_name,
            x_label=ntrg * [f"{key} value"],
            y_label=ntrg * ["predicted value"],
            close_plot=close_plot,
        )
        # plot prediction errors
        path = output_dir / f"predict_error_{key}"
        plot_data_vs_predict_error(
            plot_targets_data,
            plot_targets_pred,
            path,
            plot_name=plot_name,
            x_label=ntrg * [f"{key} value"],
            y_label=ntrg * ["prediction error"],
            close_plot=close_plot,
        )


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

    # set the device/logger pair
    device, logger = common.initialize_run(
        pathlib.Path(__file__).parent,
        pathlib.Path(__file__).stem,
        params,
    )

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
    run_evaluate(params, device, logger)


if __name__ == "__main__":
    main()
