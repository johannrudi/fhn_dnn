"""
Run Script
"""

import argparse, os, pprint, timeit, sys
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
#from torch.utils.tensorboard import SummaryWriter #TODO derprecated

###DEV###
sys.path.append('/Users/jrudi/code/dl-kit')
from dlkit.log.log_util import (logging_set_up, logging_get_logger)
from dlkit.opt.train import train_epochs

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from utils import (ModeKeys, ModelType, load_parameters, save_parameters, update_parameters_from_args)

from data import (load_data, preprocess_features, preprocess_features_noise, preprocess_labels, postprocess_labels, create_dataloader)
from model import (create_denseNet, create_convNet, create_transformerNet)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ^TODO review usage of device variable
def run(args, params):
    _prefix          = '[run_dnn]'
    mode_name        = params['runconfig']['mode']
    model_type_name  = params['model']['model_type']
    enable_verbose   = params['runconfig']['verbose']

    # set environment
    self_dir = os.path.dirname(os.path.abspath(__file__))
    mode = ModeKeys.get_from_name(mode_name)
    model_type = ModelType.get_from_name(model_type_name)

    # fix random seed for reproducibility
    if 'random_seed' in params['data'] and params['data']['random_seed'] is not None:
        np.random.seed(params['data']['random_seed'])
        torch.manual_seed(params['data']['random_seed'])
    else:
        params['data']['random_seed'] = None

    # print environment
    print(_prefix, 'Environment')
    print(_prefix, '- Directory:         ', self_dir)
    print(_prefix, '- PyTorch version:   ', torch.__version__)
    print(_prefix, '- Mode name ; key:   ', mode_name, ';', mode)
    print(_prefix, '- Model type ; key:  ', model_type_name, ';', model_type)
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
    labels_train,   labels_validate,   labels_test,   \
    features_noise_train, features_noise_validate, features_noise_test = load_data(params)

    # preprocess data
    features_scale = preprocess_features(features_train, features_test, params)
    labels_scale   = preprocess_labels  (labels_train,   labels_test,   params)
    preprocess_features_noise(features_noise_train, features_noise_test, features_scale)

    # print info about data
    logging_set_up( os.path.join(self_dir, params['runconfig']['model_dir'], _prefix.strip('[]')) )
    logger = logging_get_logger(_prefix)
    logger.info('features_train shape: {}'.format(features_train.shape))
    logger.info('features_test shape:  {}'.format(features_test.shape))
    logger.info('num_features:         {}'.format(params['model']['num_features']))
    logger.info('features scale:       {}'.format(features_scale))
    logger.info('labels_train shape:   {}'.format(labels_train.shape))
    logger.info('labels_test shape:    {}'.format(labels_test.shape))
    logger.info('num_labels:           {}'.format(params['model']['num_labels']))
    logger.info('labels scale:         {}'.format(labels_scale))
    if features_noise_train is not None and features_noise_test is not None:
        logger.info('features_noise_train shape: {}'.format(features_train.shape))
        logger.info('features_noise_test shape:  {}'.format(features_test.shape))

    # bundle arrays
    features = {
        'train':    features_train,
        'validate': features_validate,
        'test':     features_test,
    }
    targets = {
        'train':    labels_train,
        'validate': labels_validate,
        'test':     labels_test,
    }
    if features_noise_train is not None and \
       features_noise_validate is not None and \
       features_noise_test is not None:
        features_noise = {
            'train':    features_noise_train,
            'validate': features_noise_validate,
            'test':     features_noise_test,
        }
    else:
        features_noise = {
            'train':    None,
            'validate': None,
            'test':     None,
        }

    # create dataloader
    dataloader = create_dataloader(params, mode, features, targets, features_noise=features_noise)

    # create model
    if ModelType.DENSENET == model_type:
        model = create_denseNet(params)
    elif ModelType.CONVNET == model_type:
        model = create_convNet(params)
    elif ModelType.TRANSFORMERNET == model_type:
        model = create_transformerNet(params)
    else:
        raise NotImplementedError()
    print(_prefix, 'Model summary')
    print(model)

    # load model weights
    if params['runconfig']['model_load']:
        model_path = os.path.join(self_dir, params['runconfig']['model_load'])
        model.load_state_dict(torch.load(model_path))

    #
    # Training
    #

    if ModeKeys.TRAIN == mode:
        print(_prefix, 'Train')

        # create optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=params['optimizer']['learning_rate'],
                                     betas=(params['optimizer']['beta1'], params['optimizer']['beta2']),
                                     eps=params['optimizer']['epsilon'])
        loss_fn = torch.nn.MSELoss()

        # create a callback that saves the model's weights
        checkpoint_dir    = os.path.join(self_dir, params['runconfig']['model_dir'], 'checkpoints')
        checkpoint_epochs = params['runconfig']['save_checkpoints_epochs']
        #checkpoint_callback = SummaryWriter(log_dir=checkpoint_path, #save_steps=checkpoint_epochs, flush_secs=1)

        # train network
        train_log = train_epochs(
                params['training']['epochs'], model, dataloader, optimizer, loss_fn,
                device=device, logger=logger, checkpoint_epochs=checkpoint_epochs, checkpoint_dir=checkpoint_dir
        )
        time_train = train_log['time_train']
#        time_train = timeit.default_timer()
#        for epoch in tqdm(range(params['training']['epochs'])):
#            for i, data in enumerate(dataloader):
#                # set model to training mode
#                model.train()

#                # get input and target tensors
#                inputs, targets = data
#                inputs = inputs.to(device)
#                targets = targets.to(device)

#                # zero the gradients
#                optimizer.zero_grad()
#                # forward pass
#                outputs = model(inputs)

#                # calculate loss and backward pass
#                loss = loss_fn(outputs, targets)
#                loss.backward()

#                # update model parameters
#                optimizer.step()

#                # log metrics
#                if i % params['runconfig']['log_steps'] == 0:
#                    checkpoint_callback.add_scalar('loss', loss.item(), i + epoch * (
#                                params['data']['Ntrain'] // params['data']['train_batch_size']))
#        time_train = timeit.default_timer() - time_train

    #
    # Evaluation
    #

    print(_prefix, 'Evaluate')

    # create dataloader
    dataloader_eval_train = create_dataloader(params, ModeKeys.TRAIN, features, targets,
                                              features_noise=features_noise,
                                              kwargs={'shuffle':False, 'drop_last':False})
    dataloader_eval_test  = create_dataloader(params, ModeKeys.EVAL, features, targets,
                                              features_noise=features_noise,
                                              kwargs={'shuffle':False, 'drop_last':False})

    # compute predictions
    time_eval = timeit.default_timer()
    labels_train_predict, labels_test_predict = evaluate(model, dataloader_eval_train, dataloader_eval_test, params)
    time_eval = timeit.default_timer() - time_eval

    # postprocess predictions
    postprocess_labels(labels_train,         labels_test,         labels_scale)
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

def evaluate(model, dataloader_eval_train, dataloader_eval_test, params):
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        predict_list = list()
        for data in dataloader_eval_train:
            features, targets = data
            features = features.to(device)
            targets  = targets.to(device)
            predict_tensor = model(features)
            predict_list.append(predict_tensor.cpu().numpy())
        labels_train_predict = np.concatenate(predict_list, axis=0)

        predict_list = list()
        for data in dataloader_eval_test:
            features, targets = data
            features = features.to(device)
            targets  = targets.to(device)
            predict_tensor = model(features)
            predict_list.append(predict_tensor.cpu().numpy())
        labels_test_predict = np.concatenate(predict_list, axis=0)

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
        default='./configs/params.yaml',
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
