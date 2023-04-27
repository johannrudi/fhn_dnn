import torch
import torch.nn as nn
import numpy as np


def _get_activation(name):
    if 'relu' == name.casefold():
        return nn.ReLU()
    elif 'silu' == name.casefold() or 'swish' == name.casefold():
        return nn.SiLU()
    else:
        raise ValueError('Unknown name for activation function: '+name)


class create_denseNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        model_params = params['model']
        activation_fn = _get_activation(model_params['activation_fn'])
        self.flatten = nn.Flatten()  # flattens the input tensor

        # create linear layers
        self.hidden_layers = nn.ModuleList()  # list of hidden layers
        in_features = model_params['num_features'][1]
        for out_features in model_params['dense_layer_sizes']:
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(in_features, out_features),
                activation_fn,
                nn.Dropout(p=model_params['dropout']) if model_params['dropout'] else nn.Identity()
            ).apply(self.init_weights))
            in_features = out_features

        # set output layer
        self.output_layer = nn.Sequential(
            nn.Linear(in_features, model_params['num_labels']),
            nn.Sigmoid()
        ).apply(self.init_weights)

    def init_weights(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.flatten(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


def _get_conv1d_length(in_length, kernel_size, stride=1, padding=0, dilation=1):
    return int( (in_length + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1 )


class create_convNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        model_params = params['model']
        activation_fn = _get_activation(model_params['activation_fn'])
        initializer = nn.init.xavier_uniform_  # use Xavier initialization
        conv_kernel = 3
        conv_stride = 2
        conv_padding = 0
        pool_fn = nn.AvgPool1d  # AvgPool1d or MaxPool1d
        pool_kernel = 2
        pool_stride = 2
        pool_padding = 0

        # create convolutional layers
        self.hidden_conv_layers = nn.ModuleList()  # list of convolutional layers
        in_channels = model_params['num_features'][0]
        n_features = model_params['num_features'][1]
        for i, out_channels in enumerate(model_params['conv_layer_sizes']):
            if i > 0:  # apply pooling after the first convolutional layer
                self.hidden_conv_layers.append(
                        pool_fn(kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding))
                n_features = _get_conv1d_length(n_features, pool_kernel,
                                                stride=pool_stride, padding=pool_padding)
            self.hidden_conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, conv_kernel,
                          stride=conv_stride, padding=conv_padding),
                activation_fn,
                nn.Dropout(p=model_params['dropout']) if model_params['dropout'] else nn.Identity()
            ).apply(self.init_weights))
            in_channels = out_channels
            n_features = _get_conv1d_length(n_features, conv_kernel,
                                            stride=conv_stride, padding=conv_padding)

        # create linear layers
        self.flatten = nn.Flatten()  # flattens the input tensor
        self.hidden_linear_layers = nn.ModuleList()  # list of linear layers
        in_features = n_features * out_channels
        for out_features in model_params['dense_layer_sizes']:
            self.hidden_linear_layers.append(nn.Sequential(
                nn.Linear(in_features, out_features),
                activation_fn,
                nn.Dropout(p=model_params['dropout']) if model_params['dropout'] else nn.Identity()
            ).apply(self.init_weights))
            in_features = out_features

        # set output layer
        self.output_layer = nn.Sequential(
            nn.Linear(in_features, model_params['num_labels']),
            nn.Sigmoid()
        ).apply(self.init_weights)

    def init_weights(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        for layer in self.hidden_conv_layers:
            x = layer(x)
        x = self.flatten(x)
        for layer in self.hidden_linear_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x
