"""
Handling of data.
"""

import numpy as np
import os, pathlib, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from utils import ModeKeys

###############################################################################

def load_data(params, logger):
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
    elif 'RATE_DURATION'.casefold() == data_type:
        rate = np.load(data_dir/'fhn_T200_samplePrior_spikeRate.npy')
        rate = np.expand_dims(rate, axis=1)
        duration = np.load(data_dir/'fhn_T200_samplePrior_spikeDuration.npy')
        duration = np.expand_dims(duration, axis=1)
        features = np.concatenate([rate, duration], axis=1)
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    # load features noise
    if 'TIME_NOISE'.casefold() == data_type:
        features_noise = np.load(data_dir/'noise_correlated_Nt1000_Nsim10000_data.npy')
    else:
        features_noise = None

    # load labels
    labels = np.load(data_dir/'fhn_T200_samplePrior_theta.npy')

    # print info
    logger.info(f"features shape:       {features.shape}, dtype: {features.dtype}")
    logger.info(f"labels shape:         {labels.shape}, dtype: {labels.dtype}")
    if features_noise is not None:
        logger.info(f"features_noise shape: {features_noise.shape}, dtype: {features_noise.dtype}")
    assert labels.shape[0] == features.shape[0]
    assert features_noise is None or features_noise.shape == features.shape

    # reshape / expand dims
    if 2 == features.ndim:
        features = np.expand_dims(features, axis=1)
        if features_noise is not None:
            assert 2 == features_noise.ndim
            features_noise = np.expand_dims(features_noise, axis=1)

    # set sizes
    Ns = features.shape[0]
    logger.info(f"Ns:        {Ns}")
    logger.info(f"Ntrain:    {data_params['Ntrain']}")
    logger.info(f"Nvalidate: {data_params['Nvalidate']}")
    logger.info(f"Ntest:     {data_params['Ntest']}")
    assert (data_params['Ntrain'] + data_params['Nvalidate'] + data_params['Ntest']) <= Ns

    # split data into training and testing sets
    features_train    = features[:data_params['Ntrain'],...]
    labels_train      = labels  [:data_params['Ntrain'],...]
    features_validate = features[data_params['Ntrain']:data_params['Ntrain']+data_params['Nvalidate'],...]
    labels_validate   = labels  [data_params['Ntrain']:data_params['Ntrain']+data_params['Nvalidate'],...]
    features_test     = features[-data_params['Ntest']:,...]
    labels_test       = labels  [-data_params['Ntest']:,...]
    logger.info(f"features_train_shape:    {features_train.shape}")
    logger.info(f"features_validate_shape: {features_validate.shape}")
    logger.info(f"features_test_shape:     {features_test.shape}")
    logger.info(f"labels_train_shape:      {labels_train.shape}")
    logger.info(f"labels_validate_shape:   {labels_validate.shape}")
    logger.info(f"labels_test_shape:       {labels_test.shape}")
    if features_noise is not None:
        features_noise_train    = features_noise[:data_params['Ntrain'],...]
        features_noise_validate = features_noise[data_params['Ntrain']:data_params['Ntrain']+data_params['Nvalidate'],...]
        features_noise_test     = features_noise[-data_params['Ntest']:,...]
        logger.info(f"features_noise_train shape:    {features_noise_train.shape}")
        logger.info(f"features_noise_validate shape: {features_noise_validate.shape}")
        logger.info(f"features_noise_test shape:     {features_noise_test.shape}")
    else:
        features_noise_train    = None
        features_noise_validate = None
        features_noise_test     = None

    # add noise
#   if features_noise is not None:
#       features_train    += features_noise_train
#       features_validate += features_noise_validate
#       features_test     += features_noise_test

    # set dimensions
    params['data']['num_features'] = features.shape[1:]
    params['data']['num_labels']   = labels.shape[1]
    logger.debug(f"num_features: {params['data']['num_features']}")
    logger.debug(f"num_labels:   {params['data']['num_labels']}")

    # bundle arrays
    features = {
        'train':    features_train,
        'validate': features_validate,
        'test':     features_test,
    }
    labels = {
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

    return features, labels, features_noise

def _log_transform(data, shift=0.0):
    """ Applies log-transform for preprocessing. """
    if isinstance(data, dict):
        for key in data.keys():
            np.log(shift + data[key], out=data[key])
    else:
        np.log(shift + data, out=data)

def _log_transform_inverse(data, shift=0.0):
    """ Applies inverse of log-transform for postprocessing. """
    if isinstance(data, dict):
        for key in data.keys():
            np.exp(data[key], out=data[key])
            np.add(data[key], -shift, out=data[key])
    else:
        np.exp(data, out=data)
        np.add(data, -shift, out=data)

def _apply_scale(data, scale):
    """ Applies scale for preprocessing. """
    if isinstance(data, dict):
        for key in data.keys():
            np.add(     data[key], -scale['shift'],   out=data[key])
            np.multiply(data[key], 1.0/scale['mult'], out=data[key])
    else:
        np.add(     data, -scale['shift'],   out=data)
        np.multiply(data, 1.0/scale['mult'], out=data)

def _apply_scale_inverse(data, scale):
    """ Applies inverse scale for postprocessing. """
    if isinstance(data, dict):
        for key in data.keys():
            np.multiply(data[key], scale['mult'],  out=data[key])
            np.add(     data[key], scale['shift'], out=data[key])
    else:
        np.multiply(data, scale['mult'],  out=data)
        np.add(     data, scale['shift'], out=data)

def preprocess_features(features: dict, params, logger):
    data_type = params['data']['data_type'].casefold()
    # apply transformation
    if 'TIME'.casefold()       == data_type or \
       'TIME_NOISE'.casefold() == data_type or \
       'RATE'.casefold()       == data_type:
       pass
    elif 'RATE_DURATION'.casefold() == data_type:
        for key in features.keys():
            _log_transform(features[key][...,1], shift=1.0)
    else:
        raise ValueError(f"Unknown data_type: {data_type}")
    # calculate scaling
    if 'TIME'.casefold()       == data_type or \
       'TIME_NOISE'.casefold() == data_type:
        # nothing to do
        scale = {'shift': 0.0, 'mult': 1.0}
    elif 'RATE'.casefold()          == data_type or \
         'RATE_DURATION'.casefold() == data_type:
        assert 3 == features['train'].ndim
        assert 1 == features['train'].shape[1]
        features_min  = np.nanmin (features['train'], axis=0, keepdims=True)
        features_max  = np.nanmax (features['train'], axis=0, keepdims=True)
        features_mean = np.nanmean(features['train'], axis=0, keepdims=True)
        features_std  = np.nanstd (features['train'], axis=0, keepdims=True)
        # set scale
        scale = {'shift': features_min, 'mult': (features_max - features_min)}
        if 'RATE_DURATION'.casefold() == data_type:
            scale['shift'][...,1] = features_mean[...,1]
            scale['mult'][...,1]  = features_std[...,1]
    else:
        raise ValueError(f"Unknown data_type: {data_type}")
    # apply scaling
    logger.info(f"features scale: {scale}")
    _apply_scale(features, scale)
    # replace nan values
    if 'TIME'.casefold()       == data_type or \
       'TIME_NOISE'.casefold() == data_type:
       pass
    elif 'RATE'.casefold()          == data_type or \
         'RATE_DURATION'.casefold() == data_type:
        for key in features.keys():
            features[key] = np.where(np.isnan(features[key]), -1.0, features[key])
    else:
        raise ValueError(f"Unknown data_type: {data_type}")
    # return scale
    return scale

def preprocess_features_noise(features_noise, scale):
    if features_noise['train'] is not None and \
       features_noise['validate'] is not None and \
       features_noise['test'] is not None:
        # apply scaling
        _apply_scale(features_noise, scale)

def preprocess_labels(labels: dict, params, logger):
    scale = {'shift': 0.0, 'mult': 1.0}
    # apply scaling
    logger.info(f"labels scale: {scale}")
    _apply_scale(labels, scale)
    # return scale
    return scale

def postprocess_labels(labels_predict, scale):
    # apply scaling
    _apply_scale_inverse(labels_predict, scale)

###############################################################################

def _get_positions_from_histogram(data, range, n_bins, relevant_bins_threshold):
    hist, bin_edges = np.histogram(data.flatten(), range=range, bins=n_bins)
    relevant_bin_indices = hist > relevant_bins_threshold
    relevant_bin_edges   = (bin_edges[:-1])[relevant_bin_indices]
    n_relevant_bins      = np.sum(relevant_bin_indices)
    if 10 < n_relevant_bins:
        cond_positions = np.linspace(relevant_bin_edges[0], relevant_bin_edges[-1], 5)
    else:
        cond_positions = relevant_bin_edges
    return cond_positions

def get_conditional_positions(features: np.ndarray, params):
    data_type = params['data']['data_type'].casefold()
    # set function parameters
    fn_params = {
        'n_bins': {
            'TIME':     None,
            'RATE':     1000,
            'DURATION':  100,
        },
        'range': {
            'TIME':     None,
            'RATE':     [ 0.5, 1.0],
            'DURATION': [-0.5, 0.5],
        },
        'relevant_bins_threshold': {
            'TIME':     None,
            'RATE':     features.shape[0] * 0.06,
            'DURATION': features.shape[0] * 0.02,
        }
    }
    # extract conditional positions
    assert 1 < features.shape[0]
    cond_positions = list()
    if 'TIME'.casefold()       == data_type or \
       'TIME_NOISE'.casefold() == data_type:
        raise NotImplementedError()
    elif 'RATE'.casefold()          == data_type:
        i = 0
        key = 'RATE'
        cond_positions.append(_get_positions_from_histogram(
                features[...,i],
                fn_params['range'][key],
                fn_params['n_bins'][key],
                fn_params['relevant_bins_threshold'][key]
        ))
    elif 'RATE_DURATION'.casefold() == data_type:
        for i, key in enumerate(['RATE', 'DURATION']):
            cond_positions.append(_get_positions_from_histogram(
                    features[...,i],
                    fn_params['range'][key],
                    fn_params['n_bins'][key],
                    fn_params['relevant_bins_threshold'][key]
            ))
    else:
        raise ValueError(f"Unknown data_type: {data_type}")
    return cond_positions

def _filter_samples(features, targets, position, threshold):
    for i, (pos, thresh) in enumerate(zip(position, threshold)):
        features_ = features[...,i].flatten()
        idx_thresh = np.logical_and((pos - thresh) < features_, features_ < (pos + thresh))
        if 0 == i:
            indices = idx_thresh
        else:
            indices = np.logical_and(indices, idx_thresh)
    features_filtered = features[indices]
    if 2 == targets.ndim:
        targets_filtered = targets[indices]
    elif 3 == targets.ndim:
        targets_filtered = targets[:,indices,...]
    else:
        raise NotImplementedError(f"targets.ndim={targets.ndim}")
    return features_filtered, targets_filtered

def get_conditional_samples(features: np.ndarray, targets: np.ndarray, position, params):
    data_type = params['data']['data_type'].casefold()
    # extract conditional samples
    assert 1 < features.shape[0]
    if 'TIME'.casefold()       == data_type or \
       'TIME_NOISE'.casefold() == data_type:
        raise NotImplementedError()
    elif 'RATE'.casefold()          == data_type or \
         'RATE_DURATION'.casefold() == data_type:
        threshold = [0.01, 0.10]
        features_cond, targets_cond = _filter_samples(features, targets, position, threshold)
    else:
        raise ValueError(f"Unknown data_type: {data_type}")
    return features_cond, targets_cond

###############################################################################

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

class FHN_Dataset(Dataset):
    def __init__(self, features, targets, features_noise=None, item_return_order='yx'):
        super().__init__()
        self.features = torch.from_numpy(features)
        self.targets  = torch.from_numpy(targets)
        assert self.features.size(0) == self.targets.size(0)
        if features_noise is not None:
            self.features_noise = torch.from_numpy(features_noise)
        else:
            self.features_noise = None
        self.item_return_order = item_return_order.casefold()

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
        targets = self.targets[idx]

        if 'xy'.casefold() == self.item_return_order:
            sample = (targets, features)
        elif 'yx'.casefold() == self.item_return_order:
            sample = (features, targets)
        else:
            raise ValueError(f"Unknown item return order: {self.item_return_order}")
        return sample


def create_dataloader(params, logger, mode, features, targets,
                      features_noise={'train':None, 'validate':None, 'test':None, },
                      item_return_order='yx',
                      dataloader_kwargs={'shuffle':True, 'drop_last':False}):
    """ Creates a PyTorch dataset and dataloader from numpy arrays.
        Ref: https://pytorch.org/docs/stable/data.html
    """
    if ModeKeys.TRAIN == mode or \
       ModeKeys.VALIDATE == mode:
        batch_size = params['data']['train_batch_size']
    else:
        batch_size = params['data']['eval_batch_size']

    # create the dataset
    logger.info('Create new dataset')
    if ModeKeys.TRAIN == mode:
        dataset = FHN_Dataset(features['train'], targets['train'],
                              features_noise=features_noise['train'],
                              item_return_order=item_return_order)
    elif ModeKeys.VALIDATE == mode:
        dataset = FHN_Dataset(features['validate'], targets['validate'],
                              features_noise=features_noise['validate'],
                              item_return_order=item_return_order)
    else:
        dataset = FHN_Dataset(features['test'], targets['test'],
                              features_noise=features_noise['test'],
                              item_return_order=item_return_order)

    # create the dataloader
    logger.info('Create new dataloader')
    dataloader = DataLoader(dataset, batch_size=batch_size, **dataloader_kwargs)

    # output
    return dataloader

