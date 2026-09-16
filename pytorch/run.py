"""
Chain train.py then evaluate.py for a combined train/eval run.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import common
import evaluate
import train
from dlk.mgmt import parameters as config_params
from dlk.mgmt.log import logging_get_logger
from dlk.mode import Mode, get_mode_from_name


def main() -> None:
    """Parse CLI args once, set up logging once, then train and/or evaluate."""
    # <params>

    parser = argparse.ArgumentParser()
    config_params.add_args_to_parser(
        parser,
        default_params_path="configs/params_dnn.yaml",
        default_mode="train_eval",
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

    # save parameters for reproducibility
    config_params.save(params, save_dir=params["runconfig"]["save_dir"])

    # set one device/logger pair for the whole process
    device, _ = common.initialize_run(
        pathlib.Path(__file__).parent,
        pathlib.Path(__file__).stem,
        params,
    )

    # get mode
    mode = get_mode_from_name(params["runconfig"]["mode"])
    assert mode is not None

    # train the network
    if Mode.TRAIN in mode:
        train.run_train(
            params,
            device=device,
            logger=logging_get_logger("train"),
        )
        # force evaluate to auto-discover the checkpoint just written
        params["runconfig"]["load_checkpoint"] = None

    # evaluate the network
    if mode.any(Mode.PREDICT | Mode.EVAL):
        evaluate.run_evaluate(
            params,
            device=device,
            logger=logging_get_logger("evaluate"),
        )


if __name__ == "__main__":
    main()
