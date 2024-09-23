"""
Handling of Data
"""

import numpy as np
import os, pathlib, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from utils import ModeKeys


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
            print('[load_data]', 'features_noise shape:', features_noise.shape, '- dtype:', features_noise.dtype)
    assert labels.shape[0] == features.shape[0]
    assert features_noise is None or features_noise.shape == features.shape

    # reshape / expand dims
    if 2 == features.ndim:
        features = np.expand_dims(features, axis=1)
        if features_noise is not None:
            assert 2 == features_noise.ndim
            features_noise = np.expand_dims(features_noise, axis=1)

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
    if features_noise is not None:
        features_noise_train    = features_noise[:data_params['Ntrain'],...]
        features_noise_validate = features_noise[data_params['Ntrain']:data_params['Ntrain']+data_params['Nvalidate'],...]
        features_noise_test     = features_noise[-data_params['Ntest']:,...]
    else:
        features_noise_train    = None
        features_noise_validate = None
        features_noise_test     = None

    # add noise
#   if features_noise is not None:
#       features_train    += features_noise_train
#       features_validate += features_noise_validate
#       features_test     += features_noise_test

    return features_train, features_validate, features_test, \
           labels_train,   labels_validate,   labels_test, \
           features_noise_train, features_noise_validate, features_noise_test

def _preprocess_apply_scale(arrays, scale):
    for a in arrays:
        np.add(a, -scale['shift'], out=a)
        np.multiply(a, 1.0/scale['mult'], out=a)

def preprocess_features(features_train, features_test, params):
    data_type = params['data']['data_type'].casefold()
    # calculate min and max
    if 'TIME'.casefold()       == data_type or \
       'TIME_NOISE'.casefold() == data_type:
        # nothing to do
        scale = {'shift': 0.0, 'mult': 1.0}
    elif 'RATE'.casefold()          == data_type or \
         'RATE_DURATION'.casefold() == data_type:
        features_min = np.zeros((1,features_train.shape[1]))
        features_max = np.expand_dims(np.amax(features_train, axis=0), axis=0)
        scale = {'shift': features_min, 'mult': (features_max - features_min)}
    else:
        raise ValueError('Unknown parameter for data->data_type')
    # apply scaling
    _preprocess_apply_scale((features_train, features_test), scale)
    # return scale
    return scale

def preprocess_features_noise(features_noise_train, features_noise_test, scale):
    if features_noise_train is not None and features_noise_test is not None:
        # apply scaling
        _preprocess_apply_scale((features_noise_train, features_noise_test), scale)

def preprocess_labels(labels_train, labels_test, params):
    labels_min = np.expand_dims(np.amin(labels_train, axis=0), axis=0)
    labels_max = np.expand_dims(np.amax(labels_train, axis=0), axis=0)
    scale = {'shift': labels_min, 'mult': (labels_max - labels_min)}
    # apply scaling
    _preprocess_apply_scale((labels_train, labels_test), scale)
    # return scale
    return scale

def _postprocess_apply_scale(arrays, scale):
    for a in arrays:
        np.multiply(a, scale['mult'], out=a)
        np.add(a, scale['shift'], out=a)

def postprocess_labels(labels_train_predict, labels_test_predict, scale):
    # apply scaling
    _postprocess_apply_scale((labels_train_predict, labels_test_predict), scale)

###############################################################################

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

class FHN_Dataset(Dataset):
    def __init__(self, features, targets, features_noise=None):
        super().__init__()
        self.features = torch.from_numpy(features)
        self.targets  = torch.from_numpy(targets)
        assert self.features.size(0) == self.targets.size(0)
        if features_noise is not None:
            self.features_noise = torch.from_numpy(features_noise)
        else:
            self.features_noise = None

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.features_noise is not None:
            noise_idx = torch.randint(self.features_noise.size(0), (1,))[0]
            features = self.features[idx] + self.features_noise[noise_idx]
        else:
            features = self.features[idx]
        targets  = self.targets[idx]

        return (features, targets)


def create_dataloader(params, mode, features, targets,
                      features_noise={'train':None, 'validate':None, 'test':None, },
                      kwargs={'shuffle':True, 'drop_last':False}):
    """ Creates a PyTorch dataset and dataloader from numpy arrays.
        Ref: https://pytorch.org/docs/stable/data.html
    """
    enable_training = (mode == ModeKeys.TRAIN)
    enable_verbose  = params['runconfig']['verbose']
    if enable_training:
        batch_size = params['data']['train_batch_size']
    else:
        batch_size = params['data']['eval_batch_size']

    # create the dataset
    if enable_verbose: print('[create_dataloader]', 'Create new dataset')
    if enable_training:
        dataset = FHN_Dataset(features['train'], targets['train'],
                              features_noise=features_noise['train'])
    else:
        dataset = FHN_Dataset(features['test'], targets['test'],
                              features_noise=features_noise['test'])

    # create the dataloader
    if enable_verbose: print('[create_dataloader]', 'Create new dataloader')
    dataloader = DataLoader(dataset, batch_size=batch_size, **kwargs)

    # output
    return dataloader

