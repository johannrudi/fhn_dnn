"""
Handling of Data
"""

import numpy as np
import pathlib
import tensorflow

from tensorflow.data import Dataset
from tensorflow.estimator import ModeKeys

def load_data(params):
    enable_verbose = params['runconfig']['verbose']
    data_params    = params['data']
    data_type      = params['data']['data_type'].casefold()

    # set directory for data
    data_dir = pathlib.Path(data_params['data_dir'])

    # load features
    if 'TIME'.casefold()       == data_type or \
       'TIME_NOISE'.casefold() == data_type:
        features = np.load(data_dir/'fhn_T200_samplePrior_state0.npy')
    elif 'RATE'.casefold() == data_type:
        rate = np.load(data_dir/'fhn_T200_samplePrior_spikeRate.npy')
        rate = np.expand_dims(rate, axis=1)
        features = np.where(np.isnan(rate), 0.0, rate)  # set nan values to zero
    elif 'RATE_DURATION'.casefold() == data_type:
        rate = np.load(data_dir/'fhn_T200_samplePrior_spikeRate.npy')
        rate = np.expand_dims(rate, axis=1)
        rate = np.where(np.isnan(rate), 0.0, rate)  # set nan values to zero
        duration = np.load(data_dir/'fhn_T200_samplePrior_spikeDuration.npy')
        duration = np.expand_dims(duration, axis=1)
        features = np.concatenate([rate, duration], axis=1)
    else:
        raise ValueError('Unknown parameter for data->data_type')

    # load features noise
    if 'TIME_NOISE'.casefold() == data_type:
        features_noise = np.load(data_dir/'noise_correlated_Nt1000_Nsim10000_data.npy')
    else:
        features_noise = None

    # load labels
    labels = np.load(data_dir/'fhn_T200_samplePrior_theta.npy')

    # print info
    if enable_verbose:
        print('[load_data]', 'features shape:', features.shape, '- dtype:', features.dtype)
        print('[load_data]', 'labels shape:  ', labels.shape,   '- dtype:', labels.dtype)
        if features_noise is not None:
            print('[load_data]', 'features_noise shape:', features_noise.shape,   '- dtype:', features_noise.dtype)
    assert labels.shape[0] == features.shape[0]
    assert features_noise is None or features_noise.shape == features.shape

    # reshape / expand dims
    if 2 == features.ndim:
        features = np.expand_dims(features, axis=2)
        if features_noise is not None:
            assert 2 == features_noise.ndim
            features_noise = np.expand_dims(features_noise, axis=2)

    # set dimensions
    params['model']['num_features'] = features.shape[1:]
    params['model']['num_labels']   = labels.shape[1]

    # set sizes
    Ns = features.shape[0]
    if enable_verbose:
        print('[load_data]', 'Ns:', Ns)
        print('[load_data]', 'Ntrain:   ', data_params['Ntrain'])
        print('[load_data]', 'Nvalidate:', data_params['Nvalidate'])
        print('[load_data]', 'Ntest:    ', data_params['Ntest'])
    assert (data_params['Ntrain'] + data_params['Nvalidate'] + data_params['Ntest']) <= Ns
    assert data_params['Nvalidate'] == 0, 'no validation for now' #TODO

    # split data into training and testing sets
    features_train    = features[:data_params['Ntrain'],...]
    labels_train      = labels  [:data_params['Ntrain'],...]
    features_validate = features[data_params['Ntrain']:data_params['Ntrain']+data_params['Nvalidate'],...]
    labels_validate   = labels  [data_params['Ntrain']:data_params['Ntrain']+data_params['Nvalidate'],...]
    features_test     = features[-data_params['Ntest']:,...]
    labels_test       = labels  [-data_params['Ntest']:,...]

    # add noise
    if features_noise is not None:
        features_train    += features_noise[:data_params['Ntrain'],...]
        features_validate += features_noise[data_params['Ntrain']:data_params['Ntrain']+data_params['Nvalidate'],...]
        features_test     += features_noise[-data_params['Ntest']:,...]

    return features_train, features_validate, features_test, \
           labels_train,   labels_validate,   labels_test

def preprocess_features(features_train, features_test, params):
    data_type = params['data']['data_type'].casefold()
    # calculate min and max
    if 'TIME'.casefold()       == data_type or \
       'TIME_NOISE'.casefold() == data_type:
        # nothing to do
        scale = {'shift': 0.0, 'mult': 1.0}
        return features_train, features_test, scale
    elif 'RATE'.casefold()          == data_type or \
         'RATE_DURATION'.casefold() == data_type:
        features_min = np.zeros((1,features_train.shape[1]))
        features_max = np.expand_dims(np.amax(features_train, axis=0), axis=0)
    else:
        raise ValueError('Unknown parameter for data->data_type')
    # scale by min and max
    features_train = (features_train - features_min) / (features_max - features_min)
    features_test  = (features_test  - features_min) / (features_max - features_min)
    scale = {'shift': features_min, 'mult': (features_max - features_min)}
    # return features
    return features_train, features_test, scale

def preprocess_labels(labels_train, labels_test, params):
    labels_min = np.expand_dims(np.amin(labels_train, axis=0), axis=0)
    labels_max = np.expand_dims(np.amax(labels_train, axis=0), axis=0)
    labels_train = (labels_train - labels_min) / (labels_max - labels_min)
    labels_test  = (labels_test  - labels_min) / (labels_max - labels_min)
    scale = {'shift': labels_min, 'mult': (labels_max - labels_min)}
    # return labels
    return labels_train, labels_test, scale

def postprocess_labels(labels_train_predict, labels_test_predict, scale):
    labels_train_predict = labels_train_predict * scale['mult'] + scale['shift']
    labels_test_predict  = labels_test_predict * scale['mult'] + scale['shift']
    return labels_train_predict, labels_test_predict

def create_dataset(params, mode,
                   features_train, features_validate, features_test,
                   labels_train,   labels_validate,   labels_test,
                   repeat=False, prefetch=False):
    """ Creates a tensorflow dataset from numpy arrays.
        Ref: https://www.tensorflow.org/api_docs/python/tf/data/Dataset
    """
    enable_training = (mode == ModeKeys.TRAIN)
    enable_verbose  = params['runconfig']['verbose']

    # create the dataset
    if enable_verbose:  print('[create_dataset]', 'Create new dataset from tensor slices')
    if enable_training:
        dataset = Dataset.from_tensor_slices((features_train, labels_train))
    else:
        dataset = Dataset.from_tensor_slices((features_test, labels_test))

    # shuffle the dataset
    if enable_training and params['data']['train_shuffle_buffer_size']:
        bufsiz  = params['data']['train_shuffle_buffer_size']
        rndseed = params['data']['random_seed']
        if enable_verbose:  print('[create_dataset]', 'Shuffle the dataset, buffer size', bufsiz, 'random seed', rndseed)
        dataset = dataset.shuffle(bufsiz, seed=rndseed, reshuffle_each_iteration=True)

    # repeat the dataset
    if enable_training and repeat:
        if enable_verbose:  print('[create_dataset]', 'Repeat the dataset')
        dataset = dataset.repeat()

    # batch the dataset
    if enable_training:
        if enable_verbose:  print('[create_dataset]', 'Batch the dataset, batch size', params['data']['train_batch_size'])
        dataset = dataset.batch(params['data']['train_batch_size'], drop_remainder=True)
    else:
        if enable_verbose:  print('[create_dataset]', 'Batch the dataset, batch size', params['data']['eval_batch_size'])
        dataset = dataset.batch(params['data']['eval_batch_size'], drop_remainder=True)

    # activate prefetching
    if prefetch:
        if enable_verbose:  print('[create_dataset]', 'Prefetch the dataset')
        dataset = dataset.prefetch(tensorflow.data.experimental.AUTOTUNE)

    # return dataset
    return dataset

