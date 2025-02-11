"""
Create neural networks.
"""

import torch
import torch.nn as nn
import numpy as np

from dlkit.nets.mlp import (MLPModel, MLPModelMultIn)
from dlkit.nets.conv1d import Conv1dModel
from dlkit.nets.transformer import Transformer1d0dModel
from dlkit.nets.autoencoder import Autoencoder

from utils import NetworkType

###############################################################################

def _get_activation(name):
    if 'relu' == name.casefold():
        return nn.ReLU()
    elif 'gelu' == name.casefold():
        return nn.GELU()
    elif 'silu' == name.casefold() or 'swish' == name.casefold():
        return nn.SiLU()
    else:
        raise ValueError('Unknown name for activation function: '+name)

def _get_conv1d_length(in_length, kernel_size, stride=1, padding=0, dilation=1):
    return int( (in_length + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1 )

def _create_denseNet(params, logger):
    net_params = params['net']
    if params['data']['autoencoder_load_dir']:
        input_size = params['data']['autoencoder_latent_space']
    else:
        input_size = params['data']['num_features'][1]
    output_size = params['data']['num_labels']
    return MLPModel(
            input_size, output_size,
            hidden_layers_sizes=net_params['dense_layer_sizes'],
            hidden_layers_activation=_get_activation(net_params['activation_fn']),
            output_layer_activation=None,
            use_dropout=net_params['dropout']
    )

def _create_convNet(params, logger):
    net_params = params['net']
    hidden_conv_layers_kernels = len(net_params['conv_layer_sizes'])*[3]
    hidden_conv_layers_kwargs = {'stride': 2, 'padding': 0}
    # calculate length of features after convolutional layers
    n_features = params['data']['num_features'][1]
    for i, _ in enumerate(net_params['conv_layer_sizes']):
        n_features = _get_conv1d_length(n_features, hidden_conv_layers_kernels[i],
                                        stride=hidden_conv_layers_kwargs['stride'],
                                        padding=hidden_conv_layers_kwargs['padding'])
    n_channels = net_params['conv_layer_sizes'][-1] * params['data']['num_features'][0]
    n_features = n_features * n_channels
    return Conv1dModel(
            params['data']['num_features'][0], # input_channels
            hidden_conv_layers_channels_mult=net_params['conv_layer_sizes'],
            hidden_conv_layers_kernels=hidden_conv_layers_kernels,
            hidden_conv_layers_activation=_get_activation(net_params['activation_fn']),
            hidden_conv_layers_kwargs=hidden_conv_layers_kwargs,
            hidden_dense_input_size=n_features,
            hidden_dense_layers_sizes=net_params['dense_layer_sizes'],
            hidden_dense_layers_activation=_get_activation(net_params['activation_fn']),
            output_size=params['data']['num_labels'],
            output_layer_activation=None,
            use_dropout=net_params['dropout']
    )

def _create_transformerNet(params, logger):
    net_params = params['net']
    return Transformer1d0dModel(
            params['data']['num_features'][1], # src_size
            params['data']['num_labels'],      # trg_size
            net_params['transformer_embedding_size'],
            net_params['transformer_n_heads'],
            net_params['transformer_feedforward_size'],
            output_layer_activation=None,
            use_dropout=net_params['dropout']
    )

###############################################################################

def create_dnn(params, logger):
    net_params = params['net']
    net_type   = NetworkType.get_from_name(net_params['type'])
    logger.info(f"Network type: {net_params['type']}, key: {net_type}")
    # create network
    if NetworkType.DENSENET == net_type:
        net = _create_denseNet(params, logger)
    elif NetworkType.CONVNET == net_type:
        net = _create_convNet(params, logger)
    elif NetworkType.TRANSFORMERNET == net_type:
        net = _create_transformerNet(params, logger)
    else:
        raise NotImplementedError()
    # return network
    return net

def create_enc_dec(params, logger):
    e_net_params    = params['e_net']
    d_net_params    = params['d_net']
    e_net_type      = NetworkType.get_from_name(e_net_params['type'])
    d_net_type      = NetworkType.get_from_name(d_net_params['type'])
    logger.info(f"Encoder type: {e_net_params['type']}, key: {e_net_type}")
    logger.info(f"Decoder type: {d_net_params['type']}, key: {d_net_type}")
    # create encoder network
    if NetworkType.DENSENET == e_net_type:
        e_net = MLPModel(
            params['data']['num_features'][1], # input_size
            params['data']['latent_dim'],      # output_size
            hidden_layers_sizes=e_net_params['dense_layer_sizes'],
            hidden_layers_activation=_get_activation(e_net_params['activation_fn']),
            output_layer_activation=None,
            use_dropout=e_net_params['dropout']
    )
    elif NetworkType.CONVNET == e_net_type:
        hidden_conv_layers_kernels = len(e_net_params['conv_layer_sizes']) * [3]
        hidden_conv_layers_kwargs = {'stride': 2, 'padding': 0}
#       e_net = Conv1dModel(
#           params['data']['num_features'][0],  # input_channels
#           hidden_conv_layers_channels_mult=e_net_params['conv_layer_sizes'],
#           hidden_conv_layers_kernels=hidden_conv_layers_kernels,
#           hidden_conv_layers_activation=_get_activation(e_net_params['activation_fn']),
#           hidden_conv_layers_kwargs=hidden_conv_layers_kwargs,
#           use_dropout=e_net_params['dropout']
#       )
    #elif NetworkType.TRANSFORMERNET == e_net_type:
    #   TODO
    else:
        raise NotImplementedError()
    # create decoder network
    if NetworkType.DENSENET == d_net_type:
        d_net = MLPModel(
            params['data']['latent_dim'],      # input_size
            params['data']['num_features'][1], # output_size
            hidden_layers_sizes=d_net_params['dense_layer_sizes'],
            hidden_layers_activation=_get_activation(d_net_params['activation_fn']),
            output_layer_activation=None,
            use_dropout=d_net_params['dropout']
    )
    elif NetworkType.CONVNET == d_net_type:
        hidden_conv_layers_kernels = len(d_net_params['conv_layer_sizes']) * [3]
        hidden_conv_layers_kwargs = {'stride': 1, 'padding': 1, 'padding_mode': 'zeros'}
#       d_net = Conv1dUpscaleModelInterpolate(
#           params['data']['latent_dim'],  # input_size
#           hidden_conv_layers_channels_mult=d_net_params['conv_layer_sizes'],
#           hidden_conv_layers_kernels=hidden_conv_layers_kernels,
#           hidden_conv_layers_activation=_get_activation(d_net_params['activation_fn']),
#           hidden_conv_layers_kwargs=hidden_conv_layers_kwargs,
#           use_dropout=d_net_params['dropout']
#       )
    #elif NetworkType.TRANSFORMERNET == d_net_type:
    #   TODO
    else:
        raise NotImplementedError()
    # return networks
    return e_net, d_net

def create_ae(params, logger):
    e_net, d_net = create_enc_dec(params, logger)
    def _output_transf(y):
        return y.reshape(-1, *params['data']['num_features'])
    return Autoencoder(e_net, d_net, output_layer_transformation=_output_transf)

def create_gan(params, logger):
    g_net_params    = params['g_net']
    d_net_params    = params['d_net']
    g_net_type      = NetworkType.get_from_name(g_net_params['type'])
    d_net_type      = NetworkType.get_from_name(d_net_params['type'])
    logger.info(f"Generator type:     {g_net_params['type']}, key: {g_net_type}")
    logger.info(f"Discriminator type: {d_net_params['type']}, key: {d_net_type}")
    # create generator network
    if NetworkType.DENSENET == g_net_type:
        g_net = MLPModelMultIn(
            (params['data']['num_features'][1], params['data']['latent_dim']), # input_size
            params['data']['num_labels'],                                      # output_size
            hidden_layers_sizes=g_net_params['dense_layer_sizes'],
            hidden_layers_activation=_get_activation(g_net_params['activation_fn']),
            output_layer_activation=None,
            use_dropout=g_net_params['dropout']
        )
#   elif NetworkType.CONVNET == g_net_type:
#       TODO
#   elif NetworkType.TRANSFORMERNET == g_net_type:
#       TODO
    else:
        raise NotImplementedError()
    # create discriminator network
    if NetworkType.DENSENET == d_net_type:
        d_net = MLPModelMultIn(
            (params['data']['num_features'][1], params['data']['num_labels']), # input_size
            1,                                                                 # output_size
            hidden_layers_sizes=d_net_params['dense_layer_sizes'],
            hidden_layers_activation=_get_activation(d_net_params['activation_fn']),
            use_dropout=g_net_params['dropout']
        )
#   elif NetworkType.CONVNET == g_net_type:
#       TODO
#   elif NetworkType.TRANSFORMERNET == g_net_type:
#       TODO
    else:
        raise NotImplementedError()
    # return networks
    return g_net, d_net

### OLD:

#class DenseNet(nn.Module):
#    def __init__(self, params):
#        super().__init__()
#        model_params = params['model']
#        activation_fn = _get_activation(model_params['activation_fn'])
#        self.flatten = nn.Flatten()  # flattens the input tensor
#
#        # create linear layers
#        self.hidden_layers = nn.ModuleList()  # list of hidden layers
#        in_features = model_params['num_features'][1]
#        for out_features in model_params['dense_layer_sizes']:
#            self.hidden_layers.append(nn.Sequential(
#                nn.Linear(in_features, out_features),
#                activation_fn,
#                nn.Dropout(p=model_params['dropout']) if model_params['dropout'] else nn.Identity()
#            ).apply(self.init_weights))
#            in_features = out_features
#
#        # set output layer
#        self.output_layer = nn.Sequential(
#            nn.Linear(in_features, model_params['num_labels']),
#            nn.Sigmoid()
#        ).apply(self.init_weights)
#
#    def init_weights(self,m):
#        if isinstance(m, nn.Linear):
#            nn.init.xavier_uniform_(m.weight)
#
#    def forward(self, x):
#        x = self.flatten(x)
#        for layer in self.hidden_layers:
#            x = layer(x)
#        x = self.output_layer(x)
#        return x

#class ConvNet(nn.Module):
#    def __init__(self, params):
#        super().__init__()
#        model_params = params['model']
#        activation_fn = _get_activation(model_params['activation_fn'])
#        initializer = nn.init.xavier_uniform_  # use Xavier initialization
#        conv_kernel = 3
#        conv_stride = 2
#        conv_padding = 0
#        pool_fn = nn.AvgPool1d  # AvgPool1d or MaxPool1d
#        pool_kernel = 2
#        pool_stride = 2
#        pool_padding = 0
#
#        # create convolutional layers
#        self.hidden_conv_layers = nn.ModuleList()  # list of convolutional layers
#        in_channels = model_params['num_features'][0]
#        n_features = model_params['num_features'][1]
#        for i, out_channels in enumerate(model_params['conv_layer_sizes']):
#            if i > 0:  # apply pooling after the first convolutional layer
#                self.hidden_conv_layers.append(
#                        pool_fn(kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding))
#                n_features = _get_conv1d_length(n_features, pool_kernel,
#                                                stride=pool_stride, padding=pool_padding)
#            self.hidden_conv_layers.append(nn.Sequential(
#                nn.Conv1d(in_channels, out_channels, conv_kernel,
#                          stride=conv_stride, padding=conv_padding),
#                activation_fn,
#                nn.Dropout(p=model_params['dropout']) if model_params['dropout'] else nn.Identity()
#            ).apply(self.init_weights))
#            in_channels = out_channels
#            n_features = _get_conv1d_length(n_features, conv_kernel,
#                                            stride=conv_stride, padding=conv_padding)
#
#        # create linear layers
#        self.flatten = nn.Flatten()  # flattens the input tensor
#        self.hidden_linear_layers = nn.ModuleList()  # list of linear layers
#        in_features = n_features * out_channels
#        for out_features in model_params['dense_layer_sizes']:
#            self.hidden_linear_layers.append(nn.Sequential(
#                nn.Linear(in_features, out_features),
#                activation_fn,
#                nn.Dropout(p=model_params['dropout']) if model_params['dropout'] else nn.Identity()
#            ).apply(self.init_weights))
#            in_features = out_features
#
#        # set output layer
#        self.output_layer = nn.Sequential(
#            nn.Linear(in_features, model_params['num_labels']),
#            nn.Sigmoid()
#        ).apply(self.init_weights)
#
#    def init_weights(self,m):
#        if isinstance(m, nn.Linear):
#            nn.init.xavier_uniform_(m.weight)
#
#    def forward(self, x):
#        for layer in self.hidden_conv_layers:
#            x = layer(x)
#        x = self.flatten(x)
#        for layer in self.hidden_linear_layers:
#            x = layer(x)
#        x = self.output_layer(x)
#        return x
