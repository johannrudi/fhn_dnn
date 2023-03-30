import torch.nn as nn
import torch


class DenseNN(nn.Module):
    def __init__(self, params):
        super(DenseNN, self).__init__()
        model_params = params['model']
        initializer = nn.init.xavier_uniform_  # use Xavier initialization
        self.flatten = nn.Flatten()  # flatten the input tensor
        self.hidden_layers = nn.ModuleList()  # list of hidden layers

        # create dense layers
        for layer_size in model_params['dense_layer_sizes']:
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(model_params['num_features'], layer_size),
                nn.ReLU(),
                nn.Dropout(p=model_params['dropout']) if model_params['dropout'] else nn.Identity()
            ))

        # set output layer
        self.output_layer = nn.Sequential(
            nn.Linear(model_params['dense_layer_sizes'][-1], model_params['num_labels']),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


