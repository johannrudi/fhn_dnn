"""
Run Script
"""

import argparse, os, pprint, timeit, sys
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

import sklearn.metrics as metrics
import matplotlib.pyplot as plt

from data import (load_data, preprocess_features, preprocess_labels, postprocess_labels, create_dataset)
from model_pytorch import (create_denseNN, create_convNN)

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from utils import (load_parameters, save_parameters, update_parameters_from_args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def run(args, params):
    _prefix          = '[run_dnn]'
    mode_name        = params['runconfig']['mode'].casefold()
    model_type       = params['model']['model_type'].casefold()
    enable_verbose   = params['runconfig']['verbose']

    # set environment
    self_dir = os.path.dirname(os.path.abspath(__file__))
    if   'TRAIN'.casefold() == mode_name:
        mode = "train"
    elif 'EVAL'.casefold() == mode_name:
        mode = "eval"
    elif 'PREDICT'.casefold() == mode_name:
        mode = "predict"
    else:
        raise ValueError('Unknown value for mode: '+mode_name)

    # fix random seed for reproducibility
    if params['data']['random_seed'] is not None:
        np.random.seed(params['data']['random_seed'])
        torch.manual_seed(params['data']['random_seed'])

    # print environment
    print(_prefix, 'Environment')
    print(_prefix, '- Directory:         ', self_dir)
    print(_prefix, '- PyTorch version:   ', torch.__version__)
    print(_prefix, '- Mode name:         ', mode_name)
    print(_prefix, '- Mode key:          ', mode)
    print(_prefix, '- Seed:              ', params['data']['random_seed'])

    # print parameters
    if enable_verbose:
        print(_prefix, 'Parameters')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    # initialize timers
    time_train = 0.0
    time_eval  = 0.0

    #
    # Data & Model
    #

    # load data
    features_train, features_validate, features_test, \
    labels_train,   labels_validate,   labels_test  = load_data(params)

    # preprocess data
    features_train, features_test, features_scale = preprocess_features(features_train, features_test, params)
    labels_train, labels_test,     labels_scale   = preprocess_labels(labels_train, labels_test, params)

    # print info about data
    if enable_verbose:
        print(_prefix, 'features_train shape:', features_train.shape)
        print(_prefix, 'features_test shape: ', features_test.shape)
        print(_prefix, 'labels_train shape:  ', labels_train.shape)
        print(_prefix, 'labels_test shape:   ', labels_test.shape)
        print(_prefix, 'num_features:        ', params['model']['num_features'])
        print(_prefix, 'num_labels:          ', params['model']['num_labels'])
        print(_prefix, 'features scale:      ', features_scale)
        print(_prefix, 'labels scale:        ', labels_scale)

    # create dataset
    dataset = create_dataset(params, mode,
                             features_train, features_validate, features_test,
                             labels_train,   labels_validate,   labels_test)

    # create model
    if   'denseNN'.casefold() == model_type:
        model = create_denseNN(params)
    elif 'convNN'.casefold() == model_type:
        model = create_convNN(params)
    else:
        raise NotImplementedError()
    if enable_verbose:
        print(_prefix, 'Model summary')
        print(model)

    # load model weights
    if params['runconfig']['model_load']:
        model_path = os.path.join(self_dir, params['runconfig']['model_load'])
        model.load_state_dict(torch.load(model_path))

    #
    # Training
    #

    if mode == 'train':
        print(_prefix, 'Train')

        # create optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=params['optimizer']['learning_rate'],
                                     betas=(params['optimizer']['beta1'], params['optimizer']['beta2']),
                                     eps=params['optimizer']['epsilon'])
        loss_func = torch.nn.MSELoss()

        # create a callback that saves the model's weights
        checkpoint_path = os.path.join(self_dir, params['runconfig']['model_dir'], 'model.ckpt-{epoch:04d}')
        checkpoint_freq = params['runconfig']['save_checkpoints_steps'] * (
                    params['data']['Ntrain'] // params['data']['train_batch_size'])
        checkpoint_callback = torch.utils.tensorboard.SummaryWriter(log_dir=checkpoint_path, save_steps=checkpoint_freq,
                                                                    flush_secs=1)

        # fit model
        time_train = timeit.default_timer()
        for epoch in range(params['training']['epochs']):
            for i, data in enumerate(dataset):
                # set model to training mode
                model.train()

                # get input and target tensors
                inputs, targets = data
                inputs = inputs.to(device)
                targets = targets.to(device)

                # zero the gradients
                optimizer.zero_grad()
                # forward pass
                outputs = model(inputs)

                # calculate loss and backward pass
                loss = loss_func(outputs, targets)
                loss.backward()

                # update model parameters
                optimizer.step()

                # log metrics
                if i % params['runconfig']['log_steps'] == 0:
                    checkpoint_callback.add_scalar('loss', loss.item(), i + epoch * (
                                params['data']['Ntrain'] // params['data']['train_batch_size']))

        time_train = timeit.default_timer() - time_train

    #
    # Evaluation
    #

    print(_prefix, 'Evaluate')

    # compute predictions
    time_eval = timeit.default_timer()
    labels_train_predict, labels_test_predict = evaluate(features_train, labels_train,
                                                         features_test,  labels_test,
                                                         model, params)
    time_eval = timeit.default_timer() - time_eval

    # postprocess predictions
    labels_train, labels_test = \
        postprocess_labels(labels_train, labels_test, labels_scale)
    labels_train_predict, labels_test_predict = \
        postprocess_labels(labels_train_predict, labels_test_predict, labels_scale)

    # compute evaluation metrics
    r2_train = [metrics.r2_score(labels_train[:,i], labels_train_predict[:,i]) for i in range(labels_train.shape[1])]
    r2_test  = [metrics.r2_score(labels_test[:,i], labels_test_predict[:,i]) for i in range(labels_test.shape[1])]
    r2_train_all = metrics.r2_score(labels_train, labels_train_predict)
    r2_test_all  = metrics.r2_score(labels_test, labels_test_predict)

    # print metrics
    print(_prefix, 'Evaluate')
    print(_prefix, '- R2 score (train):', r2_train, r2_train_all)
    print(_prefix, '- R2 score (eval): ', r2_test,  r2_test_all)

    # print runtimes
    print(_prefix, 'Runtime [sec]')
    print(_prefix, '- train:', time_train)
    print(_prefix, '- eval: ', time_eval)
    print(_prefix, 'Runtime statistics')
    if 0 < time_train:
        n_epoch   = params['training']['epochs']
        n_steps   = params['training']['epochs'] * (params['data']['Ntrain']//params['data']['train_batch_size'])
        n_samples = params['data']['train_batch_size']
        print(_prefix, '- train - #epochs:         ', n_epoch)
        print(_prefix, '- train - #steps:          ', n_steps)
        print(_prefix, '- train - #samples (total):', n_steps*n_samples)
        print(_prefix, '- train - avg. steps/sec:  ', n_steps/time_train)
        print(_prefix, '- train - avg. samples/sec:', n_steps*n_samples/time_train)
    if 0 < time_eval:
        n_samples = (params['data']['Ntest']//params['data']['eval_batch_size']) * params['data']['eval_batch_size']
        print(_prefix, '- eval  - #samples:        ', n_samples)
        print(_prefix, '- eval  - avg. samples/sec:', n_samples/time_eval)

    # plot true training values vs. predictions
    n_plot_cols = labels_train.shape[1]
    fig, ax = plt.subplots(1, 2, figsize=(12,3))
    for i in range(n_plot_cols):
        ymin = np.amin(labels_train[:,i])
        ymax = np.amax(labels_train[:,i])
        ax[i].scatter(labels_train[:,i], labels_train_predict[:,i], s=4**2, alpha=0.5)
        ax[i].plot([ymin, ymax], [ymin, ymax], linewidth=3, linestyle='--', color='orange')
        ax[i].set_xlabel('train value')
        ax[i].set_ylabel('predicted value')
        ax[i].set_title('theta_%d'%i)
        ax[i].grid()
    fig.tight_layout()

    # plot true testing values vs. predictions
    n_plot_cols = labels_test.shape[1]
    fig, ax = plt.subplots(1, 2, figsize=(12,3))
    for i in range(n_plot_cols):
        ymin = np.amin(labels_test[:,i])
        ymax = np.amax(labels_test[:,i])
        ax[i].scatter(labels_test[:,i], labels_test_predict[:,i], s=4**2, alpha=0.5)
        ax[i].plot([ymin, ymax], [ymin, ymax], linewidth=3, linestyle='--', color='orange')
        ax[i].set_xlabel('test value')
        ax[i].set_ylabel('predicted value')
        ax[i].set_title('theta_%d'%i)
        ax[i].grid()
    fig.tight_layout()

    plt.show()

###############################################################################

def evaluate(features_train, labels_train, features_test, labels_test, model, params):
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        features_train_tensor = torch.tensor(features_train, dtype=torch.float32).to(device)
        features_test_tensor = torch.tensor(features_test, dtype=torch.float32).to(device)

        labels_train_predict_tensor = model(features_train_tensor)
        labels_test_predict_tensor = model(features_test_tensor)

        labels_train_predict = labels_train_predict_tensor.cpu().numpy()
        labels_test_predict = labels_test_predict_tensor.cpu().numpy()

    return labels_train_predict, labels_test_predict

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
        default='configs/params.yaml',
        help="Path to .yaml file with model parameters",
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
    save_parameters(params, model_dir=params['runconfig']['model_dir'])
    # run script
    run(args, params)

if __name__ == '__main__':
    main()
