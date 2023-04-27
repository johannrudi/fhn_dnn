import torch.nn as nn
import torch


class create_denseNN(nn.Module):
    def __init__(self, params):
        super(create_denseNN, self).__init__()
        model_params = params['model']
        self.flatten = nn.Flatten()  # flatten the input tensor
        self.hidden_layers = nn.ModuleList()  # list of hidden layers

        # create dense layers
        for layer_size in model_params['dense_layer_sizes']:
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(model_params['num_features'], layer_size),
                nn.ReLU(),
                nn.Dropout(p=model_params['dropout']) if model_params['dropout'] else nn.Identity()
            ).apply(self.init_weights))

        # set output layer
        self.output_layer = nn.Sequential(
            nn.Linear(model_params['dense_layer_sizes'][-1], model_params['num_labels']),
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


class create_convNN(nn.Module):
    def __init__(self, params):
        super(create_convNN, self).__init__()
        model_params = params['model']
        initializer = nn.init.xavier_uniform_  # use Xavier initialization
        conv_kernel = 3
        conv_strides = 2
        conv_padding = 'valid'  # 'valid' or 'same'
        pool_fn = nn.AvgPool1d  # AveragePooling1D or MaxPooling1D
        pool_size = 2
        pool_strides = 2
        pool_padding = 'valid'  # 'valid' or 'same'
        self.conv_layers = nn.ModuleList()  # list of convolutional layers

        # create convolutional layers
        for i, filter_size in enumerate(model_params['conv_layer_sizes']):
            if i > 0:  # apply pooling after the first convolutional layer
                self.conv_layers.append(pool_fn(kernel_size=pool_size, stride=pool_strides, padding=pool_padding))
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(model_params['num_features'], filter_size, conv_kernel, stride=conv_strides,
                          padding=conv_padding),
                nn.ReLU(),
                nn.Dropout(p=model_params['dropout']) if model_params['dropout'] else nn.Identity()
            ).apply(self.init_weights))

        # create dense layers
        self.hidden_layers = nn.ModuleList()
        num_conv_layers = len(model_params['conv_layer_sizes'])
        num_features = model_params['num_features'] // (
                    2 ** num_conv_layers)  # calculate the number of features after convolution and pooling
        for layer_size in model_params['dense_layer_sizes']:
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(num_features * model_params['conv_layer_sizes'][-1], layer_size),
                nn.ReLU(),
                nn.Dropout(p=model_params['dropout']) if model_params['dropout'] else nn.Identity()
            ).apply(self.init_weights))
            num_features = layer_size  # update the number of features for the next hidden layer

        # set output layer
        self.output_layer = nn.Sequential(
            nn.Linear(model_params['dense_layer_sizes'][-1], model_params['num_labels']),
            nn.Sigmoid()
        ).apply(self.init_weights)
    def init_weights(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.flatten(start_dim=1)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x
