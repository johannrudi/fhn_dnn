"""
Chain train.py then evaluate.py for a combined train/eval run.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import common
import dlk.mgmt.parameters as parameters
import dlk.opt.distributed as distributed
import evaluate
import train
from dlk.mgmt.log import logging_get_logger
from dlk.mode import Mode, get_mode_from_name


def main() -> None:
    """Parse CLI args once, set up logging once, then train and/or evaluate."""
    # initialize distributed parallelism (training only)
    with distributed.session() as ctx:

        # <params>

        parser = argparse.ArgumentParser()
        parameters.add_args_to_parser(
            parser,
            default_params_path="configs/params_dnn.yaml",
            default_mode="train_eval",
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
        common.initialize_run(
            pathlib.Path(__file__).parent,
            pathlib.Path(__file__).stem,
            ctx,
            params,
        )

        # get mode
        mode = get_mode_from_name(params["runconfig"]["mode"])
        assert mode is not None

        # save parameters for reproducibility
        if ctx.is_main:
            parameters.save(params, save_dir=params["runconfig"]["save_dir"])

        # train the network
        if Mode.TRAIN in mode:
            train.run_train(
                params,
                ctx,
                logger=logging_get_logger("train"),
            )
            # force evaluate to auto-discover the checkpoint just written
            params["runconfig"]["load_checkpoint"] = None

    # evaluate the network on the main process only, after the process group is
    # destroyed
    # NOTE: use `ctx.is_main` here: `distributed.is_main_process()` is True on
    #       every rank once the process group no longer exists
    if ctx.is_main and mode.any(Mode.PREDICT | Mode.EVAL):
        evaluate.run_evaluate(
            params,
            device=ctx.device,
            logger=logging_get_logger("evaluate"),
        )


if __name__ == "__main__":
    main()
