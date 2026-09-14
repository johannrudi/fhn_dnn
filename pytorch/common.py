"""
Shared setup helpers for train.py, evaluate.py, and run.py.
"""

from __future__ import annotations

import logging
import os
import pathlib
import pprint
import random
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from dlk.mgmt.log import logging_get_logger, logging_set_up
from nets import create_ae

from data import load_data, preprocess_features, preprocess_targets


def initialize_run(
    self_dir: pathlib.Path,
    self_name: str,
    params: dict[str, Any],
) -> tuple[torch.device, logging.Logger]:
    """Set compute device, random seed, and logging for one process.

    Call exactly once per process. A combined ``run.py`` invocation builds
    ``device``/``logger`` here and passes them into both ``run_train`` and
    ``run_evaluate`` so only one log-file set is created.

    Args:
        self_dir: Directory that contains the calling script (usually
            ``pathlib.Path(__file__).parent``).
        self_name: Log-file stem for this process (``"train"``,
            ``"evaluate"``, or ``"run"``).
        params: Full configuration dict. Mutates
            ``params["runconfig"]["random_seed"]`` to ``None`` when the key
            is absent, matching ``run_dnn.py``.

    Returns:
        ``(device, logger)`` pair used by the rest of the run.
    """
    enable_debug = params["runconfig"].get("debug")

    # check compute environment
    cpu_logical_cores = os.cpu_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # fix random seed for reproducibility
    random_seed = params["runconfig"].get("random_seed")
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    else:
        params["runconfig"]["random_seed"] = None

    # set up logging
    logging_set_up(self_dir / params["runconfig"]["save_dir"] / self_name)
    logger = logging_get_logger(self_name)

    # log environment (Mode / Data key are logged by each caller)
    logger.info(f"Environment - Directory:         {self_dir}")
    logger.info(f"Environment - PyTorch version:   {torch.__version__}")
    logger.info(f"Environment - CPU logical cores: {cpu_logical_cores}")
    logger.info(f"Environment - Torch device:      {device}")
    logger.info(f"Environment - Seed:              {random_seed}")

    # print parameters
    if enable_debug:
        print("<parameters>")
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)
        print("</parameters>")

    return device, logger


def load_and_preprocess_data(
    params: dict[str, Any],
    device: torch.device,
    logger: logging.Logger,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    Any,
    Any,
    Callable | None,
    Callable | None,
]:
    """Load features/targets and build optional input transforms.

    Does not create dataloaders; each caller builds its own. Behavior matches
    ``run_dnn.py`` data setup, including the optional FFT transform and the
    ``####DEV`` autoencoder-encoder path (file discovery uses ``pathlib``).

    Args:
        params: Full configuration dict.
        device: Torch device for loading an optional autoencoder checkpoint.
        logger: Logger used for autoencoder load messages (and as the parent
            context for this run).

    Returns:
        Tuple
        ``(features, targets, features_noise, targets_noise, targets_scale,
        targets_noise_scale, features_transform_fn, train_input_transform_fn)``.
    """
    # load data
    features, targets, features_noise, targets_noise = load_data(
        params, logging_get_logger("load_data")
    )

    # preprocess data
    features_scale = preprocess_features(
        features, params, logging_get_logger("preprocess_features")
    )
    targets_scale = preprocess_targets(
        targets, params, logging_get_logger("preprocess_targets")
    )
    preprocess_features(
        features_noise,
        params,
        logging_get_logger("preprocess_features_noise"),
        scale=features_scale,
        array_name="features_noise",
    )
    targets_noise_scale = preprocess_targets(
        targets_noise,
        params,
        logging_get_logger("preprocess_targets_noise"),
        array_name="targets_noise",
    )

    # set transform functions
    features_transform_fn: Callable[[torch.Tensor], torch.Tensor] | None = None
    train_input_transform_fn: Callable[[torch.Tensor], torch.Tensor] | None = None

    if params["data"].get("features_fft"):

        def _fft_transform(features: torch.Tensor) -> torch.Tensor:
            # subsample features
            features_half = features[..., ::2]
            size = features_half.size()

            # compute FFT (use rfft for real-valued input)
            features_fft = torch.fft.rfft(features, dim=-1, norm="ortho")
            features_fft = features_fft[..., : size[-1]]

            # concatenate features and FFT real and imag parts
            features_transformed = torch.concatenate(
                (features_half, features_fft.real, features_fft.imag), dim=1
            )
            return features_transformed

        features_transform_fn = _fft_transform

    ####DEV
    if params["data"].get("autoencoder_load_dir"):
        import yaml

        autoencoder_load_dir = pathlib.Path(params["data"]["autoencoder_load_dir"])
        requested_param_file = autoencoder_load_dir / "params.yaml"

        checkpoint_folders = list((autoencoder_load_dir / "checkpoints").glob("*"))
        latest_folder = max(checkpoint_folders, key=lambda p: p.stat().st_mtime)
        checkpoint_files = list(latest_folder.glob("*.pt"))
        requested_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)

        logger.info(f"Load autoencoder: use parameter file: {requested_param_file}")
        logger.info(f"Load autoencoder: use checkpoint file: {requested_checkpoint}")

        # load the AE params
        with open(requested_param_file, "r") as file:
            ae_params = yaml.safe_load(file)
        ae_params["data"]["num_features"] = params["data"]["num_features"]

        # load the AE network
        autoencoder = create_ae(ae_params, logging_get_logger("create_autoencoder"))
        checkpoint = torch.load(requested_checkpoint, map_location=device)
        autoencoder.load_state_dict(checkpoint["model_state_dict"])
        autoencoder.to(device)
        autoencoder.eval()

        print("<autoencoder>")
        print(autoencoder)
        print("</autoencoder>")

        train_input_transform_fn = autoencoder.e_net
    ####/DEV

    return (
        features,
        targets,
        features_noise,
        targets_noise,
        targets_scale,
        targets_noise_scale,
        features_transform_fn,
        train_input_transform_fn,
    )


def find_all_checkpoints(
    checkpoint_root: pathlib.Path, pattern: str = "*.pt"
) -> list[pathlib.Path]:
    """Return matching checkpoints under ``checkpoint_root/checkpoints/<latest>``.

    Picks the most recently modified timestamp subfolder under
    ``checkpoints/``, then returns files matching ``pattern`` inside it
    sorted by mtime ascending (oldest / lowest-epoch first).

    Args:
        checkpoint_root: Run directory that contains a ``checkpoints/``
            subfolder (usually ``self_dir / runconfig.save_dir``).
        pattern: Glob relative to the latest timestamp folder. Default
            ``"*.pt"``.

    Returns:
        Checkpoint paths sorted by ``st_mtime`` ascending. Never empty.

    Raises:
        FileNotFoundError: If no checkpoint folder or matching file exists.
    """
    checkpoints_dir = checkpoint_root / "checkpoints"
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(
            f"No checkpoints directory found at {checkpoints_dir}. "
            "Run train.py first, or set runconfig.load_checkpoint to a .pt file."
        )

    checkpoint_folders = [p for p in checkpoints_dir.glob("*") if p.is_dir()]
    if not checkpoint_folders:
        raise FileNotFoundError(
            f"No checkpoint folders found under {checkpoints_dir}. "
            "Run train.py first, or set runconfig.load_checkpoint to a .pt file."
        )

    latest_folder = max(checkpoint_folders, key=lambda p: p.stat().st_mtime)
    checkpoint_files = list(latest_folder.glob(pattern))
    if not checkpoint_files:
        raise FileNotFoundError(
            f"No checkpoint files matching {pattern!r} found under {latest_folder}. "
            "Run train.py first, or set runconfig.load_checkpoint to a .pt file."
        )

    return sorted(checkpoint_files, key=lambda p: p.stat().st_mtime)


def find_latest_checkpoint(
    checkpoint_root: pathlib.Path, pattern: str = "*.pt"
) -> pathlib.Path:
    """Return the newest matching checkpoint under ``checkpoint_root``.

    Thin wrapper around ``find_all_checkpoints``: returns the last entry
    (highest mtime) of that list. Within one stream selected by
    ``pattern``, that is the highest-epoch file.

    Args:
        checkpoint_root: Run directory that contains a ``checkpoints/``
            subfolder (usually ``self_dir / runconfig.save_dir``).
        pattern: Glob forwarded to ``find_all_checkpoints``. Default
            ``"*.pt"``.

    Returns:
        Path to the selected checkpoint file.

    Raises:
        FileNotFoundError: If no checkpoint folder or matching file exists.
    """
    return find_all_checkpoints(checkpoint_root, pattern=pattern)[-1]
