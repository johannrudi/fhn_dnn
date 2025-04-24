"""
Run training and evaluation of autoencoder.
"""

import argparse, os, pprint, random, sys, timeit
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from dlkit.log.log_util import (logging_set_up, logging_get_logger)
from dlkit.opt.train import train_epochs

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from utils import (
    ModeKeys,
    load_parameters,
    save_parameters,
    update_parameters_from_args,
    plot_loss,
    plot_data_vs_predict,
    plot_data_vs_predict_error
)
from data import (
    dictarray_is_not_none,
    load_data,
    load_timesteps,
    preprocess_features,
    postprocess_features,
    create_dataloader
)
from nets import create_ae

###############################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ^TODO review usage of device variable

def run(args, params):
    # get parameters
    mode_name     = params['runconfig']['mode']
    enable_debug  = params['runconfig']['debug']

    # set environment
    self_dir = os.path.dirname(os.path.abspath(__file__))
    mode     = ModeKeys.get_from_name(mode_name)

    # set up logging
    logging_set_up( os.path.join(self_dir, params['runconfig']['save_dir'], 'run_dnn') )
    logger = logging_get_logger('run_dnn')

    # print environment
    logger.info(f"Environment - Directory:       {self_dir}")
    logger.info(f"Environment - PyTorch version: {torch.__version__}")
    logger.info(f"Environment - Seed:            {params['data']['random_seed']}")
    logger.info(f"Environment - Mode name:       {mode_name}, key: {mode}")

    # print parameters
    if enable_debug:
        print('<parameters>')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)
        print('</parameters>')

    # fix random seed for reproducibility
    if 'random_seed' in params['data'] and params['data']['random_seed'] is not None:
        random.seed(params['data']['random_seed'])
        np.random.seed(params['data']['random_seed'])
        torch.manual_seed(params['data']['random_seed'])
    else:
        params['data']['random_seed'] = None

    # initialize timers
    time_train = 0.0
    time_eval  = 0.0

    #
    # Data
    #

    # load data
    features, _, features_noise, _ = load_data(params, logging_get_logger('load_data'))

    # preprocess data
    features_scale = preprocess_features(features, params, logging_get_logger('preprocess_features'))
    features_noise_scale = preprocess_features(features_noise, params,
                                               logging_get_logger('preprocess_features_noise'),
                                               scale=features_scale, array_name='features_noise')

    # create dataloader
    if ModeKeys.TRAIN == mode:
        mode_to_data_key = 'train'
    elif ModeKeys.VALIDATE == mode:
        mode_to_data_key = 'validate'
    elif ModeKeys.EVAL == mode:
        mode_to_data_key = 'test'
    else:
        raise NotImplementedError()
    dataloader = create_dataloader(
            params,
            logging_get_logger('create_dataloader'),
            mode,
            features          = features[mode_to_data_key],
            targets           = None,
            features_noise    = features_noise[mode_to_data_key],
            targets_noise     = None,
            item_return_order = 'yy'
    )

    #
    # Network
    #

    # create network
    net = create_ae(params, logging_get_logger('create_network'))
    print('<network>')
    print(net)
    print('</network>')

    # load network weights
    if params['runconfig']['load_dir']:
        net_path = os.path.join(self_dir, params['runconfig']['load_dir'])
        net.load_state_dict(torch.load(net_path, map_location=device))

    # transfer to device
    net.to(device)

    #
    # Training
    #

    if ModeKeys.TRAIN == mode:
        print('<train>')

        # create optimizer
        if 'Adam'.casefold() == params['optimizer']['type'].casefold():
            optimizer = torch.optim.Adam(
                    net.parameters(),
                    lr    = params['optimizer']['learning_rate'],
                    betas = (params['optimizer']['beta1'], params['optimizer']['beta2']),
                    eps   = params['optimizer']['epsilon']
            )
        elif 'AdamW'.casefold() == params['optimizer']['type'].casefold():
            optimizer = torch.optim.AdamW(
                    net.parameters(),
                    lr           = params['optimizer']['learning_rate'],
                    betas        = (params['optimizer']['beta1'], params['optimizer']['beta2']),
                    eps          = params['optimizer']['epsilon'],
                    weight_decay = params['optimizer']['weight_decay']
            )
        else:
            raise ValueError('Unknown name for optimizer: '+params['optimizer']['type'])

        # create learning rate scheduler
        if 'learning_rate_scheduler' in params['optimizer'] and \
           params['optimizer']['learning_rate_scheduler'] is not None:
            params_lr_scheduler = params['optimizer']['learning_rate_scheduler']
            milestone_epochs = [
                params_lr_scheduler['linear_epochs'],
                params_lr_scheduler['linear_epochs'] + params_lr_scheduler['constant_epochs'],
            ]
            schedulers = [
                torch.optim.lr_scheduler.LinearLR(
                        optimizer,
                        start_factor = params_lr_scheduler['init_learning_rate']/params['optimizer']['learning_rate'],
                        end_factor   = 1.0,
                        total_iters  = params_lr_scheduler['linear_epochs']
                ),
                torch.optim.lr_scheduler.ConstantLR(
                        optimizer,
                        factor       = 1.0,
                        total_iters  = params_lr_scheduler['constant_epochs']
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max        = params['training']['epochs'] - milestone_epochs[-1] - 1,
                        eta_min      = params_lr_scheduler['final_learning_rate']
                ),
            ]
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers = schedulers,
                    milestones = milestone_epochs
            )
        else:
            lr_scheduler = None

        # set loss function
        loss_fn = torch.nn.MSELoss()

        # checkpointing for saving network weights
        checkpoint_dir    = os.path.join(self_dir, params['runconfig']['save_dir'], 'checkpoints')
        checkpoint_epochs = params['runconfig']['save_checkpoints_epochs']

        # train network
        epoch_dlog = train_epochs(
                params['training']['epochs'],
                net,
                dataloader,
                optimizer,
                loss_fn,
                lr_scheduler        = lr_scheduler,
                device              = device,
                inputs_transform_fn = None,
                logger              = logger,
                checkpoint_epochs   = checkpoint_epochs,
                checkpoint_dir      = checkpoint_dir
        )
        time_train = epoch_dlog['time_train']

        print('</train>')

    #
    # Evaluation
    #

    print('<evaluate>')

    # create dataloaders
    eval_dataloader = dict()
    for key in features.keys():
        eval_dataloader[key] = create_dataloader(
                params,
                logging_get_logger('create_dataloader'),
                ModeKeys.EVAL,
                features          = features[key],
                targets           = None,
                features_noise    = features_noise[key],
                targets_noise     = None,
                item_return_order = 'yy'
        )

    # compute predictions
    time_eval = timeit.default_timer()
    eval_features_pred, eval_features_data = predict(net, eval_dataloader, params)
    time_eval = timeit.default_timer() - time_eval

    # postprocess evaluation data
    if dictarray_is_not_none(features):
        postprocess_features(eval_features_data, features_scale, params)
        postprocess_features(eval_features_pred, features_scale, params)
    elif dictarray_is_not_none(features_noise):
        postprocess_features(eval_features_data, features_noise_scale, params)
        postprocess_features(eval_features_pred, features_noise_scale, params)
    else:
        raise NotImplementedError()

    # compute percentiles
    percentiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    eval_features_data_percentiles, eval_features_pred_percentiles, _ = get_mse_percentiles(
            eval_features_data, eval_features_pred, percentiles)

    # compute evaluation metrics
    eval_mse, eval_mae, eval_mape, eval_medae, eval_r2 = eval_data_vs_pred(eval_features_data, eval_features_pred)
    for key in eval_features_data.keys():
        logger.info(f"Evaluate - {key} - MSE:      {eval_mse[key]}")
        logger.info(f"Evaluate - {key} - MAE:      {eval_mae[key]}")
        logger.info(f"Evaluate - {key} - MedAE:    {eval_medae[key]}")
        logger.info(f"Evaluate - {key} - MAPE:     {eval_mape[key]}")
        logger.info(f"Evaluate - {key} - R2 score: {eval_r2[key]}")

    print('</evaluate>')

    #
    # Output
    #

    # print runtimes
    logger.info(f"Runtime - train [sec]: {time_train}")
    logger.info(f"Runtime - eval [sec]:  {time_eval}")
    if 0 < time_train:
        n_epoch   = params['training']['epochs']
        n_steps   = params['training']['epochs'] * (params['data']['Ntrain']//params['data']['train_batch_size'])
        n_samples = params['data']['train_batch_size']
        logger.info(f"Runtime statistics - train - #epochs:          {n_epoch}")
        logger.info(f"Runtime statistics - train - #steps:           {n_steps}")
        logger.info(f"Runtime statistics - train - #samples (total): {n_steps*n_samples}")
        logger.info(f"Runtime statistics - train - avg. steps/sec:   {n_steps/time_train}")
        logger.info(f"Runtime statistics - train - avg. samples/sec: {n_steps*n_samples/time_train}")
    if 0 < time_eval:
        n_samples = (params['data']['Ntest']//params['data']['eval_batch_size']) * params['data']['eval_batch_size']
        logger.info(f"Runtime statistics - eval  - #samples:         {n_samples}")
        logger.info(f"Runtime statistics - eval  - avg. samples/sec: {n_samples/time_eval}")

    # plot loss
    path = os.path.join(self_dir, params['runconfig']['save_dir'], 'loss')
    plot_loss(epoch_dlog['loss_mean'], path, 'Training loss', params['training']['epochs'],
              loss_std=epoch_dlog['loss_std'], x_offset=1, y_scale='log')

    # plot predictions percentiles
    for key in eval_features_data.keys():
        # skip if no samples exist
        if params['data']['N'+key] <= 0:
            continue
        # set up plotting
        m = len(percentiles)
        fig, ax = plt.subplots(m, 1, figsize=(10, 2*m))
        y_lim = [ min(np.min(eval_features_data_percentiles[key]), np.min(eval_features_pred_percentiles[key])),
                  max(np.max(eval_features_data_percentiles[key]), np.max(eval_features_pred_percentiles[key])) ]
        # plot true values vs. predictions
        for i in range(m):
            y_data = eval_features_data_percentiles[key][i]
            y_pred = eval_features_pred_percentiles[key][i]
            y_mse  = np.mean((y_pred - y_data)**2)
            ax[i].plot(y_data, label=f"data ({key})",
                       color='tab:orange', linewidth=0, marker='.', markersize=6)
            ax[i].plot(y_pred, label=f"MSE={y_mse:.3e}",
                       color='tab:blue', linewidth=2)
            ax[i].set_ylim(y_lim)
            ax[i].set_ylabel(f"{int(percentiles[i]*100):d}%")
            ax[i].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True)
            ax[i].grid()
        ax[-1].set_xlabel('time step')
        fig.tight_layout()
        path = os.path.join(self_dir, params['runconfig']['save_dir'], 'predict_mse_percentiles_'+key)
        fig.savefig(f"{path}.pdf", dpi=300)

    # show plots
    if params['runconfig']['show_plots']:
        plt.show()

###############################################################################

def predict(net, eval_dataloader, params):
    net.eval()
    # get network predictions
    data = dict()
    pred = dict()
    with torch.no_grad():
        for key in eval_dataloader.keys():
            d_list = list()
            p_list = list()
            for x, yd in eval_dataloader[key]:
                x = x.to(device)
                yp = net(x)
                d_list.append(yd.cpu().numpy())
                p_list.append(yp.cpu().numpy())
            data[key] = np.concatenate(d_list, axis=0)
            pred[key] = np.concatenate(p_list, axis=0)
    # return predictions and (true) data
    return pred, data

def eval_data_vs_pred(data, pred):
    eval_mse   = dict()
    eval_mae   = dict()
    eval_mape  = dict()
    eval_medae = dict()
    eval_r2    = dict()
    for key in data.keys():
        data_ = data[key].squeeze()
        pred_ = pred[key].squeeze()
        eval_mse[key]   = metrics.mean_squared_error(data_, pred_)
        eval_mae[key]   = metrics.mean_absolute_error(data_, pred_)
        eval_mape[key]  = metrics.mean_absolute_percentage_error(data_, pred_)
        eval_medae[key] = metrics.median_absolute_error(data_, pred_)
        eval_r2[key]    = metrics.r2_score(data_, pred_)
    return eval_mse, eval_mae, eval_mape, eval_medae, eval_r2

def get_mse_percentiles(features_data, features_pred, percentiles):
    features_percentiles         = dict()
    features_predict_percentiles = dict()
    for key in features_data.keys():
        assert key in features_pred
        features_shape = features_data[key].shape
        # calculate MSE
        y_data = features_data[key].squeeze()
        y_pred = features_pred[key].squeeze()
        y_mse  = np.mean((y_pred - y_data)**2, axis=1)
        # sort MSE
        idx_sorted = np.argsort(y_mse)
        # get percentiles
        features_percentiles[key]         = np.empty([len(percentiles), features_shape[-1]])
        features_predict_percentiles[key] = np.empty_like(features_percentiles[key])
        for j, p in enumerate(percentiles):
            i = int(np.round(p * features_shape[0]))
            features_percentiles[key][j]         = features_data[key][idx_sorted[i]]
            features_predict_percentiles[key][j] = features_pred[key][idx_sorted[i]]
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
        default='./configs/params_ae.yaml',
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
    args   = parser.parse_args(sys.argv[1:])
    # load parameters, and save them for reproducibility
    params = load_parameters(args.params)
    update_parameters_from_args(params['runconfig'], args)
    save_parameters(params, save_dir=params['runconfig']['save_dir'])
    # run script
    run(args, params)

if __name__ == '__main__':
    main()
