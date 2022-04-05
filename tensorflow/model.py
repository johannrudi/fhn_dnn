"""
Model Definitions
"""

from tensorflow.keras import (layers, Input, Model)

def create_denseNN(params, name='denseNN', **kwargs):
    model_params = params['model']
    # setup parameters of hidden layers
    initializer  = 'glorot_uniform'
    # set input layer(s)
    inputs = Input(shape=model_params['num_features'])
    x = layers.Flatten()(inputs)
    # create dense layers
    for layer_size in model_params['dense_layer_sizes']:
        x = layers.Dense(layer_size, kernel_initializer=initializer)(x)
        x = layers.Activation(model_params['activation_fn'])(x)
        if model_params['dropout']:
            x = layers.Dropout(model_params['dropout'])(x)
    # set output layer
    x = layers.Dense(model_params['num_labels'], kernel_initializer=initializer)(x)
    outputs = layers.Activation('sigmoid')(x)
    # create model
    return Model(inputs, outputs, name=name, **kwargs)

def create_convNN(params, name='convNN', **kwargs):
    model_params = params['model']
    # setup parameters of hidden layers
    initializer  = 'glorot_uniform'
    conv_kernel  = 3
    conv_strides = 2
    conv_padding = 'valid'  # 'valid' or 'same'
    pool_fn      = layers.AveragePooling1D  # AveragePooling1D or MaxPooling1D
    pool_size    = 2
    pool_strides = 2
    pool_padding = 'valid'  # 'valid' or 'same'
    # set input layer(s)
    inputs = Input(shape=model_params['num_features'])
    # create convolutional layers
    x = inputs
    for i, filter_size in enumerate(model_params['conv_layer_sizes']):
        if 0 < i:  x = pool_fn(pool_size, strides=pool_strides, padding=pool_padding)(x)
        x = layers.Conv1D(filter_size, conv_kernel,
                          strides=conv_strides, padding=conv_padding,
                          kernel_initializer=initializer)(x)
        x = layers.Activation(model_params['activation_fn'])(x)
        if model_params['dropout']:
            x = layers.Dropout(model_params['dropout'])(x)
    # create dense layers
    x = layers.Flatten()(x)
    for layer_size in model_params['dense_layer_sizes']:
        x = layers.Dense(layer_size, kernel_initializer=initializer)(x)
        x = layers.Activation(model_params['activation_fn'])(x)
        if model_params['dropout']:
            x = layers.Dropout(model_params['dropout'])(x)
    # set output layer
    x = layers.Dense(model_params['num_labels'], kernel_initializer=initializer)(x)
    outputs = layers.Activation('sigmoid')(x)
    # create model
    return Model(inputs, outputs, name=name, **kwargs)

