"""
Run training and evaluation of DNN-based inverse map.
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
    preprocess_features,
    preprocess_targets, postprocess_targets,
    preprocess_features_noise,
    preprocess_targets_noise, postprocess_targets_noise,
    create_dataloader
)
from nets import create_dnn, create_ae

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
    features, targets, features_noise, targets_noise = load_data(params, logging_get_logger('load_data'))

    # preprocess data
    features_scale = preprocess_features(features, params, logging_get_logger('preprocess_features'))
    targets_scale  = preprocess_targets (targets,  params, logging_get_logger('preprocess_targets'))
    features_noise_scale = preprocess_features_noise(features_noise, params, logging_get_logger('preprocess_features_noise'), features_scale)
    targets_noise_scale  = preprocess_targets_noise (targets_noise,  params, logging_get_logger('preprocess_targets_noise'))

####DEV
    if params['data']['autoencoder_load_dir']:
        import glob
        import yaml

        requested_param_file = os.path.join(params['data']['autoencoder_load_dir'], 'params.yaml')

        checkpoint_folders = glob.glob(os.path.join(params['data']['autoencoder_load_dir'], "checkpoints", "*"))
        latest_folder = max(checkpoint_folders, key=os.path.getmtime)
        checkpoint_files = glob.glob(os.path.join(latest_folder, "*.pt"))
        requested_checkpoint = max(checkpoint_files, key=os.path.getmtime)

        logger.info(f"Load autoencoder: use parameter file: {requested_param_file}")
        logger.info(f"Load autoencoder: use checkpoint file: {requested_checkpoint}")

        # load the AE params
        with open(requested_param_file, 'r') as file:
            ae_params = yaml.safe_load(file)
        ae_params['data']['num_features'] = params['data']['num_features']

        # load the AE network
        autoencoder = create_ae(ae_params, logging_get_logger('create_autoencoder'))
        checkpoint  = torch.load(requested_checkpoint, map_location=device)
        autoencoder.load_state_dict(checkpoint['model_state_dict'])
        autoencoder.to(device)
        autoencoder.eval()

        print('<autoencoder>')
        print(autoencoder)
        print('</autoencoder>')

        features_transform_fn = autoencoder.e_net
    else:
        features_transform_fn = None
####/DEV

    # create dataloader
    dataloader = create_dataloader(
            params,
            logging_get_logger('create_dataloader'),
            mode,
            features          = features,
            targets           = targets,
            features_noise    = features_noise,
            targets_noise     = targets_noise,
            item_return_order = 'yx'
    )

    #
    # Network
    #

    # create network
    net = create_dnn(params, logging_get_logger('create_network'))
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
                inputs_transform_fn = features_transform_fn,
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
    dataloader_eval = dict()
    for key, dl_mode in zip(['train', 'validate', 'test'], [ModeKeys.TRAIN, ModeKeys.VALIDATE, ModeKeys.EVAL]):
        dataloader_eval[key] = create_dataloader(
                params,
                logging_get_logger('create_dataloader'),
                dl_mode,
                features          = features,
                targets           = targets,
                features_noise    = features_noise,
                targets_noise     = targets_noise,
                item_return_order = 'yx',
                dataloader_kwargs = {'shuffle':False, 'drop_last':False}
        )

    # compute predictions
    time_eval = timeit.default_timer()
    targets_predict = evaluate(net, dataloader_eval, params, features_transform_fn=features_transform_fn)
    time_eval = timeit.default_timer() - time_eval

    # set up evaluation data
    if dictarray_is_not_none(targets_noise):
        targets_data = {}
        for key in ['train', 'validate', 'test']:
            targets_data[key] = np.concatenate((targets[key], targets_noise[key]), axis=1)
    else:
        targets_data = targets

    # postprocess predictions
    n_targets = targets['train'].shape[1]
    for key in ['train', 'validate', 'test']:
        postprocess_targets(targets_data[key][:,:n_targets],    targets_scale)
        postprocess_targets(targets_predict[key][:,:n_targets], targets_scale)
    if dictarray_is_not_none(targets_noise):
        for key in ['train', 'validate', 'test']:
            postprocess_targets_noise(targets_data[key][:,:n_targets],    targets_noise_scale)
            postprocess_targets_noise(targets_predict[key][:,:n_targets], targets_noise_scale)

    # compute evaluation metrics
    eval_mse = dict()
    eval_r2  = dict()
    for key in ['train', 'validate', 'test']:
        y_data = targets_data[key]
        y_pred = targets_predict[key]
        eval_mse[key]        = [metrics.mean_squared_error(y_data[:,i], y_pred[:,i]) for i in range(y_data.shape[1])]
        eval_mse[key+'_all'] =  metrics.mean_squared_error(y_data, y_pred)
        eval_r2[key]        = [metrics.r2_score(y_data[:,i], y_pred[:,i]) for i in range(y_data.shape[1])]
        eval_r2[key+'_all'] =  metrics.r2_score(y_data, y_pred)
        logger.info(f"Evaluate - MSE ({key}):      " + str(eval_mse[key]) + f" {eval_mse[key+'_all']}")
        logger.info(f"Evaluate - R2 score ({key}): " + str(eval_r2[key]) + f" {eval_r2[key+'_all']}")

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

    # plot predictions
    for key in ['train', 'validate', 'test']:
        if params['data']['N'+key] <= 0:
            continue
        n_targets_data       = targets_data[key].shape[1]
        targets_plot         = [targets_data[key][:,i]    for i in range(n_targets_data)]
        targets_predict_plot = [targets_predict[key][:,i] for i in range(n_targets_data)]
        # plot true values vs. predictions
        path = os.path.join(self_dir, params['runconfig']['save_dir'], 'data_vs_predict_'+key)
        plot_data_vs_predict(
                targets_plot, targets_predict_plot, path,
                plot_name=[f"param_{i}" for i in range(n_targets_data)],
                x_label=n_targets_data*[f"{key} value"],
                y_label=n_targets_data*[f"predicted value"])
        # plot prediction errors
        path = os.path.join(self_dir, params['runconfig']['save_dir'], 'predict_error_'+key)
        plot_data_vs_predict_error(
                targets_plot, targets_predict_plot, path,
                plot_name=[f"param_{i}" for i in range(n_targets_data)],
                x_label=n_targets_data*[f"{key} value"],
                y_label=n_targets_data*[f"prediction error"])

    # show plots
    if params['runconfig']['show_plots']:
        plt.show()

###############################################################################

def evaluate(net, dataloader_eval, params, features_transform_fn=None):
    net.eval()
    # evaluate network predictions
    y_predict = dict()
    with torch.no_grad():
        for key in dataloader_eval.keys():
            y_list = list()
            for data in dataloader_eval[key]:
                x, _ = data
                x = x.to(device)
                if features_transform_fn is not None:
                    x = features_transform_fn(x)
                y = net(x)
                y_list.append(y.cpu().numpy())
            y_predict[key] = np.concatenate(y_list, axis=0)
    # return predictions
    return y_predict

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
        default='./configs/params_dnn.yaml',
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
