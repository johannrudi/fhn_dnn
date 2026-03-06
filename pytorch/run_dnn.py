"""
Run training and evaluation of DNN-based inverse map.
"""

import argparse
import os
import pathlib
import pprint
import random
import sys
import timeit

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import torch
from dlk.log.log_util import logging_get_logger, logging_set_up
from dlk.nets.util import get_parameters
from dlk.opt.train import train_epochs
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from nets import create_ae, create_network
from opt_utils import create_lr_scheduler, create_optimizer

from data import (create_dataloader, dictarray_is_not_none, load_data,
                  postprocess_targets, preprocess_features, preprocess_targets)
from utils.utils import (Mode, load_parameters, plot_data_vs_predict,
                         plot_data_vs_predict_error, plot_loss, save_parameters,
                         update_parameters_from_args)

###############################################################################


def run(args, params):
    # set environment
    self_dir = os.path.dirname(os.path.abspath(__file__))
    enable_debug = params["runconfig"].get("debug")

    # check compute environment
    cpu_logical_cores = os.cpu_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set mode
    mode_name = params["runconfig"]["mode"]
    mode = None
    for name in mode_name.split("_"):
        m = Mode[name.upper()]
        mode = m if mode is None else mode | m

    # set key for data
    if mode is not None and Mode.TRAIN in mode:
        mode_to_data_key = "train"
    elif mode is not None and mode.any(Mode.PREDICT | Mode.EVAL):
        mode_to_data_key = "test"
    else:
        raise NotImplementedError()

    # set up logging
    logging_set_up(os.path.join(self_dir, params["runconfig"]["save_dir"], "run_dnn"))
    logger = logging_get_logger("run_dnn")

    # log environment
    logger.info(f"Environment - Directory:       {self_dir}")
    logger.info(f"Environment - PyTorch version: {torch.__version__}")
    logger.info(f"Environment - Seed:            {params['data']['random_seed']}")
    logger.info(f"Environment - Mode:            {mode} (--mode {mode_name})")
    logger.info(f"Environment - Data key:        {mode_to_data_key}")
    logger.info(f"Environment - CPU logical cores: {cpu_logical_cores}")
    logger.info(f"Environment - Torch device:      {device}")
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

    #
    # Data
    #

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
    features_transform_fn = None  # used in dataloader
    train_input_transform_fn = None  # used in training loops

    if params["data"].get("features_fft"):

        def features_transform_fn(features):
            # subsample features
            features_half = features[..., ::2]
            size = features_half.size()

            # compute FFT (use rfft for real-valued input)
            features_fft = torch.fft.rfft(features, dim=-1, norm="ortho")
            features_fft = features_fft[..., : size[-1]]

            # concatenate features and FFT real and imag parts
            features_transformed = torch.concatenate(
                (features_half, features_fft.real, features_fft.imag), axis=1
            )
            return features_transformed

    ####DEV
    if params["data"].get("autoencoder_load_dir"):
        import glob

        import yaml

        requested_param_file = os.path.join(
            params["data"]["autoencoder_load_dir"], "params.yaml"
        )

        checkpoint_folders = glob.glob(
            os.path.join(params["data"]["autoencoder_load_dir"], "checkpoints", "*")
        )
        latest_folder = max(checkpoint_folders, key=os.path.getmtime)
        checkpoint_files = glob.glob(os.path.join(latest_folder, "*.pt"))
        requested_checkpoint = max(checkpoint_files, key=os.path.getmtime)

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

    # create dataloader
    dataloader = create_dataloader(
        params,
        logging_get_logger("create_dataloader"),
        mode,
        features=features[mode_to_data_key],
        targets=targets[mode_to_data_key],
        features_noise=features_noise[mode_to_data_key],
        targets_noise=targets_noise[mode_to_data_key],
        features_transform_fn=features_transform_fn,
    )

    #
    # Network
    #

    # create network
    net = create_network(params, logging_get_logger("create_network"))

    # log network and parameters
    _, _, net_params_table = get_parameters(net)
    net_out_path = os.path.join(self_dir, params["runconfig"]["save_dir"], "net.txt")
    net_out = f"<network>\n{net}\n</network>\n"
    net_out += f"<parameters>\n{net_params_table}\n</parameters>\n"
    with open(net_out_path, "w") as f:
        f.write(net_out)
    if enable_debug:
        print(net_out)

    # load network weights
    if params["runconfig"]["load_dir"]:
        net_path = os.path.join(self_dir, params["runconfig"]["load_dir"])
        net.load_state_dict(torch.load(net_path, map_location=device))

    # transfer to device
    net.to(device)

    #
    # Training
    #

    if Mode.TRAIN in mode:
        # create optimizer
        optimizer = create_optimizer(net, params["optimizer"])

        # create learning rate scheduler
        lr_scheduler = create_lr_scheduler(
            optimizer, params["optimizer"], params["training"]["epochs"]
        )

        # set loss function
        loss_fn = torch.nn.MSELoss()

        # checkpointing for saving network weights
        checkpoint_dir = os.path.join(
            self_dir, params["runconfig"]["save_dir"], "checkpoints"
        )
        checkpoint_epochs = params["runconfig"]["save_checkpoints_epochs"]

        if Mode.PROFILE in mode:
            # profile training
            from dlk.opt.profiler import profile_train_batches
            from dlk.opt.train import train_batches

            train_batches_kwargs = dict(
                device=device,
                inputs_transform_fn=train_input_transform_fn,
            )
            log_profile_dir = os.path.join(
                self_dir, params["runconfig"]["save_dir"], "profile"
            )

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
            print(f"<train>")
            train_dlog = train_epochs(
                params["training"]["epochs"],
                net,
                dataloader,
                optimizer,
                loss_fn,
                lr_scheduler=lr_scheduler,
                device=device,
                inputs_transform_fn=train_input_transform_fn,
                checkpoint_epochs=checkpoint_epochs,
                checkpoint_dir=checkpoint_dir,
            )
            time_train = train_dlog.get("time_train")
            print(f"</train>")

    #
    # Prediction
    #

    if not mode.any(Mode.PREDICT | Mode.EVAL):
        return

    print("<predict>")

    # create dataloaders
    eval_dataloader = dict()
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

    # compute predictions
    time_eval = timeit.default_timer()
    eval_targets_pred, eval_targets_data = predict(
        net,
        eval_dataloader,
        params,
        device,
        input_transform_fn=train_input_transform_fn,
    )
    time_eval = timeit.default_timer() - time_eval

    # postprocess evaluation data
    if dictarray_is_not_none(targets) and dictarray_is_not_none(targets_noise):
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

    #
    # Evaluation
    #

    if Mode.EVAL not in mode:
        return

    print("<evaluate>")

    # compute evaluation metrics
    eval_mse, eval_mae, eval_r2 = eval_data_vs_pred(
        eval_targets_data, eval_targets_pred
    )
    for key in eval_targets_data.keys():
        logger.info(
            f"Evaluate - MSE ({key}):      "
            + str(eval_mse[key + "_i"])
            + f" {eval_mse[key]}"
        )
        logger.info(
            f"Evaluate - MAE ({key}):      "
            + str(eval_mae[key + "_i"])
            + f" {eval_mae[key]}"
        )
        logger.info(
            f"Evaluate - R2 score ({key}): "
            + str(eval_r2[key + "_i"])
            + f" {eval_r2[key]}"
        )

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
        train_dlog["loss_mean"],
        path,
        "Training loss",
        params["training"]["epochs"],
        loss_std=train_dlog["loss_std"],
        x_offset=1,
        y_scale="log",
    )

    # plot predictions
    for key in eval_targets_data.keys():
        # skip if no samples exist
        if params["data"]["N" + key] <= 0:
            continue
        # set up plotting
        assert eval_targets_data[key].shape[1] == eval_targets_pred[key].shape[1]
        ntrg = eval_targets_data[key].shape[1]
        plot_targets_data = [eval_targets_data[key][:, i] for i in range(ntrg)]
        plot_targets_pred = [eval_targets_pred[key][:, i] for i in range(ntrg)]
        plot_name = [f"param_{i}" for i in range(ntrg)]
        # plot true values vs. predictions
        path = os.path.join(
            self_dir, params["runconfig"]["save_dir"], "data_vs_predict_" + key
        )
        plot_data_vs_predict(
            plot_targets_data,
            plot_targets_pred,
            path,
            plot_name=plot_name,
            x_label=ntrg * [f"{key} value"],
            y_label=ntrg * [f"predicted value"],
        )
        # plot prediction errors
        path = os.path.join(
            self_dir, params["runconfig"]["save_dir"], "predict_error_" + key
        )
        plot_data_vs_predict_error(
            plot_targets_data,
            plot_targets_pred,
            path,
            plot_name=plot_name,
            x_label=ntrg * [f"{key} value"],
            y_label=ntrg * [f"prediction error"],
        )

    # show plots
    if params["runconfig"]["show_plots"]:
        plt.show()


###############################################################################


def predict(net, eval_dataloader, params, device, input_transform_fn=None):
    net.eval()
    # get network predictions
    data = dict()
    pred = dict()
    with torch.no_grad():
        for key in eval_dataloader.keys():
            d_list = list()
            p_list = list()
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


def eval_data_vs_pred(data, pred):
    eval_mse = dict()
    eval_mae = dict()
    eval_r2 = dict()
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
        default="./configs/params_dnn.yaml",
        help="Path to .yaml file with parameters",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=[
            "train",
            "predict",
            "eval",
            "train_eval",
            "train_predict",
            "train_profile",
        ],
        default="train_eval",
        help=(
            "Can train, predict, eval, and combine train_eval (default)."
            + "  eval runs on available checkpoints."
            + "  train_eval runs train, predict, and eval."
            + "  train_profile runs profiling of a few training steps."
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
