"""
Run training and evaluation of UNet.
"""

import argparse
import os
import pprint
import random
import sys
import timeit

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import torch
from dlk.log.log_util import logging_get_logger, logging_set_up
from dlk.opt.train import train_epochs
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../utils"))
from nets import create_unet

from data import (create_dataloader, load_data, load_timesteps, postprocess_features,
                  postprocess_targets, preprocess_features, preprocess_features_noise,
                  preprocess_targets)
from utils import (ModeKeys, load_parameters, plot_data_vs_predict,
                   plot_data_vs_predict_error, plot_loss, save_parameters,
                   update_parameters_from_args)

###############################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ^TODO review usage of device variable


def run(args, params):
    # get parameters
    mode_name = params["runconfig"]["mode"]
    enable_debug = params["runconfig"]["debug"]

    # set environment
    self_dir = os.path.dirname(os.path.abspath(__file__))
    mode = ModeKeys.get_from_name(mode_name)

    # set up logging
    logging_set_up(os.path.join(self_dir, params["runconfig"]["save_dir"], "run_dnn"))
    logger = logging_get_logger("run_dnn")

    # print environment
    logger.info(f"Environment - Directory:       {self_dir}")
    logger.info(f"Environment - PyTorch version: {torch.__version__}")
    logger.info(f"Environment - Seed:            {params['data']['random_seed']}")
    logger.info(f"Environment - Mode name:       {mode_name}, key: {mode}")

    # print parameters
    if enable_debug:
        print("<parameters>")
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)
        print("</parameters>")

    # fix random seed for reproducibility
    if "random_seed" in params["data"] and params["data"]["random_seed"] is not None:
        random.seed(params["data"]["random_seed"])
        np.random.seed(params["data"]["random_seed"])
        torch.manual_seed(params["data"]["random_seed"])
    else:
        params["data"]["random_seed"] = None

    # initialize timers
    time_train = 0.0
    time_eval = 0.0

    #
    # Data
    #

    # load data
    features, targets, features_noise = load_data(
        params, logging_get_logger("load_data")
    )

    # preprocess data
    features_scale = preprocess_features(
        features, params, logging_get_logger("preprocess_features")
    )
    targets_scale = preprocess_targets(
        targets, params, logging_get_logger("preprocess_targets")
    )
    preprocess_features_noise(features_noise, features_scale)

    # create dataloader
    dataloader = create_dataloader(
        params,
        logging_get_logger("create_dataloader"),
        mode,
        features,
        targets,
        features_noise=features_noise,
        item_return_order="yy",
    )

    #
    # Network
    #

    # create network
    net = create_unet(params, logging_get_logger("create_network"))
    print("<network>")
    print(net)
    print("</network>")

    # load network weights
    if params["runconfig"]["load_dir"]:
        net_path = os.path.join(self_dir, params["runconfig"]["load_dir"])
        net.load_state_dict(torch.load(net_path, map_location=device))

    # transfer to device
    net.to(device)

    #
    # Training
    #

    if ModeKeys.TRAIN == mode:
        print("<train>")

        # create optimizer
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=params["optimizer"]["learning_rate"],
            betas=(params["optimizer"]["beta1"], params["optimizer"]["beta2"]),
            eps=params["optimizer"]["epsilon"],
        )

        # set loss function
        loss_fn = torch.nn.MSELoss()

        # checkpointing for saving network weights
        checkpoint_dir = os.path.join(
            self_dir, params["runconfig"]["save_dir"], "checkpoints"
        )
        checkpoint_epochs = params["runconfig"]["save_checkpoints_epochs"]

        # train network
        epoch_dlog = train_epochs(
            params["training"]["epochs"],
            net,
            dataloader,
            optimizer,
            loss_fn,
            device=device,
            logger=logger,
            checkpoint_epochs=checkpoint_epochs,
            checkpoint_dir=checkpoint_dir,
        )
        time_train = epoch_dlog["time_train"]

        print("</train>")

    #
    # Evaluation
    #

    print("<evaluate>")

    # create dataloaders
    dataloader_eval = dict()
    for key, dl_mode in zip(
        ["train", "validate", "test"],
        [ModeKeys.TRAIN, ModeKeys.VALIDATE, ModeKeys.EVAL],
    ):
        dataloader_eval[key] = create_dataloader(
            params,
            logging_get_logger("create_dataloader"),
            dl_mode,
            features,
            targets,
            features_noise=features_noise,
            item_return_order="yy",
            dataloader_kwargs={"shuffle": False, "drop_last": False},
        )

    # compute predictions
    time_eval = timeit.default_timer()
    features_predict = evaluate(net, dataloader_eval, params)
    time_eval = timeit.default_timer() - time_eval

    # postprocess predictions
    postprocess_features(features, features_scale, params)
    postprocess_features(features_predict, features_scale, params)

    # compute percentiles
    percentiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    features_percentiles, features_predict_percentiles, _ = get_mse_percentiles(
        features, features_predict, percentiles
    )

    # compute evaluation metrics
    eval_mse = dict()
    eval_mae = dict()
    eval_r2 = dict()
    for key in ["train", "validate", "test"]:
        y_data = features[key].squeeze()
        y_pred = features_predict[key].squeeze()
        eval_mse[key] = metrics.mean_squared_error(y_data, y_pred)
        eval_mae[key] = metrics.mean_absolute_error(y_data, y_pred)
        eval_r2[key] = metrics.r2_score(y_data, y_pred)
        logger.info(f"Evaluate - MSE ({key}):      {eval_mse[key]}")
        logger.info(f"Evaluate - MAE ({key}):      {eval_mae[key]}")
        logger.info(f"Evaluate - R2 score ({key}): {eval_r2[key]}")

    print("</evaluate>")

    #
    # Output
    #

    # print runtimes
    logger.info(f"Runtime - train [sec]: {time_train}")
    logger.info(f"Runtime - eval [sec]:  {time_eval}")
    if 0 < time_train:
        n_epoch = params["training"]["epochs"]
        n_steps = params["training"]["epochs"] * (
            params["data"]["Ntrain"] // params["data"]["train_batch_size"]
        )
        n_samples = params["data"]["train_batch_size"]
        logger.info(f"Runtime statistics - train - #epochs:          {n_epoch}")
        logger.info(f"Runtime statistics - train - #steps:           {n_steps}")
        logger.info(
            f"Runtime statistics - train - #samples (total): {n_steps*n_samples}"
        )
        logger.info(
            f"Runtime statistics - train - avg. steps/sec:   {n_steps/time_train}"
        )
        logger.info(
            f"Runtime statistics - train - avg. samples/sec: {n_steps*n_samples/time_train}"
        )
    if 0 < time_eval:
        n_samples = (
            params["data"]["Ntest"] // params["data"]["eval_batch_size"]
        ) * params["data"]["eval_batch_size"]
        logger.info(f"Runtime statistics - eval  - #samples:         {n_samples}")
        logger.info(
            f"Runtime statistics - eval  - avg. samples/sec: {n_samples/time_eval}"
        )

    # plot loss
    path = os.path.join(self_dir, params["runconfig"]["save_dir"], "loss")
    plot_loss(
        epoch_dlog["loss_mean"],
        path,
        "Training loss",
        params["training"]["epochs"],
        loss_std=epoch_dlog["loss_std"],
        x_offset=1,
        y_scale="log",
    )

    # plot predictions percentiles
    timesteps = load_timesteps(params)
    for key in ["train", "validate", "test"]:
        m = len(percentiles)
        fig, ax = plt.subplots(m, 1, figsize=(10, 2 * m))
        y_lim = [
            min(
                np.min(features_percentiles[key]),
                np.min(features_predict_percentiles[key]),
            ),
            max(
                np.max(features_percentiles[key]),
                np.max(features_predict_percentiles[key]),
            ),
        ]
        for i in range(m):
            y_data = features_percentiles[key][i]
            y_pred = features_predict_percentiles[key][i]
            y_mse = np.mean((y_pred - y_data) ** 2)
            ax[i].plot(
                timesteps,
                y_data,
                label=f"data ({key})",
                color="tab:orange",
                linewidth=0,
                marker=".",
                markersize=6,
            )
            ax[i].plot(
                timesteps,
                y_pred,
                label=f"MSE={y_mse:.3e}",
                color="tab:blue",
                linewidth=2,
            )
            ax[i].set_xlim([timesteps[0], timesteps[-1]])
            ax[i].set_ylim(y_lim)
            ax[i].set_ylabel(f"{int(percentiles[i]*100):d}%")
            ax[i].legend(loc="center left", bbox_to_anchor=(1, 0.5), fancybox=True)
            ax[i].grid()
        ax[-1].set_xlabel("time [milliseconds]")
        fig.tight_layout()
        path = os.path.join(
            self_dir, params["runconfig"]["save_dir"], "predict_mse_percentiles_" + key
        )
        fig.savefig(f"{path}.pdf", dpi=300)

    # show plots
    if params["runconfig"]["show_plots"]:
        plt.show()


###############################################################################


def evaluate(net, dataloader_eval, params):
    net.eval()
    # evaluate network predictions
    y_predict = dict()
    with torch.no_grad():
        for key in dataloader_eval.keys():
            y_list = list()
            for data in dataloader_eval[key]:
                x, y = data
                x = x.to(device)
                y = y.to(device)
                y = net(x)
                y_list.append(y.cpu().numpy())
            y_predict[key] = np.concatenate(y_list, axis=0)
    # return predictions
    return y_predict


def get_mse_percentiles(features, features_predict, percentiles):
    features_percentiles = dict()
    features_predict_percentiles = dict()
    for key in features.keys():
        assert key in features_predict
        features_shape = features[key].shape
        # calculate MSE
        y_data = features[key].squeeze()
        y_pred = features_predict[key].squeeze()
        y_mse = np.mean((y_pred - y_data) ** 2, axis=1)
        # sort MSE
        idx_sorted = np.argsort(y_mse)
        # get percentiles
        features_percentiles[key] = np.empty([len(percentiles), features_shape[-1]])
        features_predict_percentiles[key] = np.empty_like(features_percentiles[key])
        for j, p in enumerate(percentiles):
            i = int(np.round(p * features_shape[0]))
            features_percentiles[key][j] = features[key][idx_sorted[i]]
            features_predict_percentiles[key][j] = features_predict[key][idx_sorted[i]]
    # return percentiles
    return features_percentiles, features_predict_percentiles, idx_sorted


###############################################################################


def create_arg_parser():
    """
    Create parser for command line args.

    :returns: ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--params",
        default="./configs/params_unet.yaml",
        help="Path to .yaml file with parameters",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["train", "eval", "eval_all", "train_and_eval"],
        default="train",
        help=(
            "Can train, eval, eval_all, or train_and_eval."
            + "  eval_all runs eval for all available checkpoints."
        ),
    )
    return parser


def main():
    # get arguments
    parser = create_arg_parser()
    args = parser.parse_args(sys.argv[1:])
    # load parameters, and save them for reproducibility
    params = load_parameters(args.params)
    update_parameters_from_args(params["runconfig"], args)
    save_parameters(params, save_dir=params["runconfig"]["save_dir"])
    # run script
    run(args, params)


if __name__ == "__main__":
    main()
