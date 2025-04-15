"""
Handling of data.
"""

import numpy as np
import os, pathlib, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from utils import ModeKeys

###############################################################################

def dictarray_empty():
    return {'train': None, 'validate': None, 'test': None}

def dictarray_set(arr_train, arr_validate, arr_test):
    return {'train': arr_train, 'validate': arr_validate, 'test': arr_test}

def dictarray_is_none(arr):
    return arr             is None or \
           arr['train']    is None or \
           arr['validate'] is None or \
           arr['test']     is None

def dictarray_is_not_none(arr):
    return not dictarray_is_none(arr)

###############################################################################

def _load_numpy(data_dir, features_type, targets_type):
    # TODO split in train, validate..
    if '2020' in data_dir.name:
        # load features
        if features_type in ['TIME'.casefold(), 'TIME_NOISE'.casefold()]:
            features = np.load(data_dir/'fhn_T200_samplePrior_state0.npy')
            features = np.expand_dims(features, axis=1)
        elif features_type == 'RATE_DURATION'.casefold():
            rate = np.load(data_dir/'fhn_T200_samplePrior_spikeRate.npy')
            rate = np.expand_dims(rate, axis=1)
            duration = np.load(data_dir/'fhn_T200_samplePrior_spikeDuration.npy')
            duration = np.expand_dims(duration, axis=1)
            features = np.concatenate([rate, duration], axis=1)
            features = np.expand_dims(features, axis=1)
        elif features_type == 'NOISE'.casefold():
            features = None
        else:
            raise ValueError(f"Unknown features_type: {features_type}")
        # load features noise
        if features_type in ['TIME_NOISE'.casefold(), 'NOISE'.casefold()]:
            features_noise = np.load(data_dir/'noise_correlated_Nt1000_Nsim10000_data.npy')
            features_noise = np.expand_dims(features_noise, axis=1)
        elif features_type in ['TIME'.casefold(), 'RATE_DURATION'.casefold()]:
            features_noise = None
        else:
            raise ValueError(f"Unknown features_type: {features_type}")
        # load targets
        if targets_type in ['ODE'.casefold(), 'ODE_NOISE'.casefold()]:
            targets = np.load(data_dir/'fhn_T200_samplePrior_theta.npy')
        elif targets_type == 'NOISE'.casefold():
            targets = None
        else:
            raise ValueError(f"Unknown targets_type: {targets_type}")
        # load targets of noise
        if targets_type in ['NOISE'.casefold(), 'ODE_NOISE'.casefold()]:
            noise_correl  = np.load(data_dir/'noise_correlated_Nt1000_Nsim10000_correlation.npy')
            noise_stddev  = np.load(data_dir/'noise_correlated_Nt1000_Nsim10000_stddev.npy')
            targets_noise = np.stack((noise_correl, noise_stddev), axis=1)
        elif targets_type == 'ODE'.casefold():
            targets_noise = None
        else:
            raise ValueError(f"Unknown targets_type: {targets_type}")
    elif '2025' in data_dir.name:
        # load features
        if features_type in ['TIME'.casefold(), 'TIME_NOISE'.casefold()]:
            state_train = np.load(data_dir/'fhn_Ntrain20000_state_Nt2000_dt0.2.npy')
            state_test  = np.load(data_dir/'fhn_Ntest2000_state_Nt2000_dt0.2.npy')
            features    = np.concatenate((state_train[:,0:1,:], state_test[:,0:1,:]), axis=0)
        elif features_type == 'RATE_DURATION'.casefold():
            state_stats_train = np.load(data_dir/'fhn_Ntrain20000_state_stats.npy')
            state_stats_test  = np.load(data_dir/'fhn_Ntest2000_state_stats.npy')
            features = np.concatenate((state_stats_train, state_stats_test), axis=0)
            features = np.expand_dims(features, axis=1)
        elif features_type == 'NOISE'.casefold():
            features = None
        else:
            raise ValueError(f"Unknown features_type: {features_type}")
        # load features of noise
        if features_type in ['NOISE'.casefold(), 'TIME_NOISE'.casefold()]:
            noise_train = np.load(data_dir/'ar1_Ntrain20000_state_Nt2000_dt0.2.npy')
            noise_test  = np.load(data_dir/'ar1_Ntest2000_state_Nt2000_dt0.2.npy')
            features_noise = np.concatenate((noise_train, noise_test), axis=0)
            features_noise = np.expand_dims(features_noise, axis=1)
        elif features_type in ['TIME'.casefold(), 'RATE_DURATION'.casefold()]:
            features_noise = None
        else:
            raise ValueError(f"Unknown features_type: {features_type}")
        # load targets
        if targets_type in ['ODE'.casefold(), 'ODE_NOISE'.casefold()]:
            param_train = np.load(data_dir/'fhn_Ntrain20000_param.npy')
            param_test  = np.load(data_dir/'fhn_Ntest2000_param.npy')
            targets     = np.concatenate((param_train, param_test), axis=0)
        elif targets_type == 'NOISE'.casefold():
            targets     = None
        else:
            raise ValueError(f"Unknown targets_type: {targets_type}")
        # load targets of noise
        if targets_type in ['NOISE'.casefold(), 'ODE_NOISE'.casefold()]:
            noise_param_train = np.load(data_dir/'ar1_Ntrain20000_param.npy')
            noise_param_test  = np.load(data_dir/'ar1_Ntest2000_param.npy')
            targets_noise     = np.concatenate((noise_param_train, noise_param_test), axis=0)
        elif targets_type == 'ODE'.casefold():
            targets_noise     = None
        else:
            raise ValueError(f"Unknown targets_type: {targets_type}")
    else:
        raise NotImplementedError(f"Unsupported data directory: {data_dir}")
    # return features and targets
    return features, targets, features_noise, targets_noise

def load_data(params, logger):
    data_params   = params['data']
    data_dir      = pathlib.Path(data_params['data_dir'])
    features_type = data_params['features_type'].casefold()
    targets_type  = data_params['targets_type'].casefold()

    # load numpy files
    features, targets, features_noise, targets_noise = _load_numpy(data_dir, features_type, targets_type)

    # print info
    logger.info(f"features shape:       {features.shape}, dtype: {features.dtype}")
    logger.info(f"targets shape:        {targets.shape}, dtype: {targets.dtype}")
    if features_noise is not None:
        logger.info(f"features_noise shape: {features_noise.shape}, dtype: {features_noise.dtype}")
    if targets_noise is not None:
        logger.info(f"targets_noise shape:  {targets_noise.shape}, dtype: {targets_noise.dtype}")
    assert targets.shape[0] == features.shape[0]
    assert features_noise is None or features_noise.shape == features.shape
    assert targets_noise  is None or features_noise is not None
    assert targets_noise  is None or targets_noise.shape[0] == features_noise.shape[0]

    # set sizes
    Ns = features.shape[0]
    logger.info(f"Ns:        {Ns}")
    logger.info(f"Ntrain:    {data_params['Ntrain']}")
    logger.info(f"Nvalidate: {data_params['Nvalidate']}")
    logger.info(f"Ntest:     {data_params['Ntest']}")
    assert (data_params['Ntrain'] + data_params['Nvalidate'] + data_params['Ntest']) <= Ns

    # split data into training and testing sets
    features_train    = features[:data_params['Ntrain'],...]
    targets_train     = targets [:data_params['Ntrain'],...]
    features_validate = features[data_params['Ntrain']:data_params['Ntrain']+data_params['Nvalidate'],...]
    targets_validate  = targets [data_params['Ntrain']:data_params['Ntrain']+data_params['Nvalidate'],...]
    features_test     = features[-data_params['Ntest']:,...]
    targets_test      = targets [-data_params['Ntest']:,...]
    logger.info(f"features_train_shape:    {features_train.shape}")
    logger.info(f"features_validate_shape: {features_validate.shape}")
    logger.info(f"features_test_shape:     {features_test.shape}")
    logger.info(f"targets_train_shape:     {targets_train.shape}")
    logger.info(f"targets_validate_shape:  {targets_validate.shape}")
    logger.info(f"targets_test_shape:      {targets_test.shape}")
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
    if targets_noise is not None:
        targets_noise_train    = targets_noise[:data_params['Ntrain'],...]
        targets_noise_validate = targets_noise[data_params['Ntrain']:data_params['Ntrain']+data_params['Nvalidate'],...]
        targets_noise_test     = targets_noise[-data_params['Ntest']:,...]
        logger.info(f"targets_noise_train shape:    {targets_noise_train.shape}")
        logger.info(f"targets_noise_validate shape: {targets_noise_validate.shape}")
        logger.info(f"targets_noise_test shape:     {targets_noise_test.shape}")
    else:
        targets_noise_train    = None
        targets_noise_validate = None
        targets_noise_test     = None

    # set dimensions
    params['data']['num_features'] = list(features.shape[1:])
    params['data']['num_targets']  = targets.shape[1]
    if params['data']['features_sub_length'] and \
       params['data']['features_sub_length'] < params['data']['num_features'][-1]:
        params['data']['num_features'][-1] = params['data']['features_sub_length']
    if targets_noise is not None:
        params['data']['num_targets'] += targets_noise.shape[1]
    logger.debug(f"num_features: {params['data']['num_features']}")
    logger.debug(f"num_targets:  {params['data']['num_targets']}")

    # bundle arrays
    if features is not None:
        features = dictarray_set(features_train, features_validate, features_test)
    else:
        features = dictarray_empty()
    if targets is not None:
        targets  = dictarray_set(targets_train, targets_validate, targets_test)
    else:
        targets  = dictarray_empty()
    if features_noise is not None:
        features_noise = dictarray_set(features_noise_train, features_noise_validate, features_noise_test)
    else:
        features_noise = dictarray_empty()
    if targets_noise is not None:
        targets_noise  = dictarray_set(targets_noise_train, targets_noise_validate, targets_noise_test)
    else:
        targets_noise  = dictarray_empty()

    return features, targets, features_noise, targets_noise

def load_timesteps(params):
    data_params = params['data']
    data_dir    = pathlib.Path(data_params['data_dir'])
    timesteps   = np.load(data_dir/'fhn_T200_samplePrior_time.npy')
    return timesteps

###############################################################################

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
    # exit if nothing to do
    if features is None or (isinstance(features, dict) and dictarray_is_none(features)):
        return None
    features_type = params['data']['features_type'].casefold()
    # apply transformation
    if 'TIME'.casefold()       == features_type or \
       'TIME_NOISE'.casefold() == features_type:
        pass
    elif 'RATE_DURATION'.casefold() == features_type:
        for key in features.keys():
            _log_transform(features[key][...,1], shift=1.0)
    else:
        raise ValueError(f"Unknown features_type: {features_type}")
    # calculate scaling
    if 'TIME'.casefold()       == features_type or \
       'TIME_NOISE'.casefold() == features_type:
        # nothing to do
        scale = {'shift': 0.0, 'mult': 1.0}
    elif 'RATE_DURATION'.casefold() == features_type:
        assert 3 == features['train'].ndim
        assert 1 == features['train'].shape[1]
        features_min  = np.nanmin (features['train'], axis=0, keepdims=True)
        features_max  = np.nanmax (features['train'], axis=0, keepdims=True)
        features_mean = np.nanmean(features['train'], axis=0, keepdims=True)
        features_std  = np.nanstd (features['train'], axis=0, keepdims=True)
        # set scale
        scale = {'shift': features_min, 'mult': (features_max - features_min)}
        if 'RATE_DURATION'.casefold() == features_type:
            scale['shift'][...,1] = features_mean[...,1]
            scale['mult'][...,1]  = features_std[...,1]
    else:
        raise ValueError(f"Unknown features_type: {features_type}")
    # apply scaling
    logger.info(f"features scale: {scale}")
    _apply_scale(features, scale)
    # replace nan values
    if 'TIME'.casefold()       == features_type or \
       'TIME_NOISE'.casefold() == features_type:
       pass
    elif 'RATE_DURATION'.casefold() == features_type:
        for key in features.keys():
            features[key] = np.where(np.isnan(features[key]), -1.0, features[key])
    else:
        raise ValueError(f"Unknown features_type: {features_type}")
    # return scale
    return scale

def postprocess_features(features_predict, scale, params):
    # exit if nothing to do
    if features_predict is None or (isinstance(features_predict, dict) and dictarray_is_none(features_predict)):
        return
    features_type = params['data']['features_type'].casefold()
    if 'TIME'.casefold()       == features_type or \
       'TIME_NOISE'.casefold() == features_type:
        # apply inverse scaling
        _apply_scale_inverse(features_predict, scale)
#   elif 'RATE_DURATION'.casefold() == features_type:
#       TODO
    else:
        raise ValueError(f"Unknown features_type: {features_type}")

def _preprocess_array(arr, arr_name, params, logger, scale=None):
    # exit if nothing to do
    if arr is None or (isinstance(arr, dict) and dictarray_is_none(arr)):
        return None
    # set scaling to no-op.
    if scale is None:
        scale = {'shift': 0.0, 'mult': 1.0}
    # apply scaling
    logger.info(f"{arr_name} scale: {scale}")
    _apply_scale(arr, scale)
    # return scale
    return scale

def _postprocess_array(arr, scale):
    # exit if nothing to do
    if arr is None or (isinstance(arr, dict) and dictarray_is_none(arr)):
        return
    # apply inverse scaling
    _apply_scale_inverse(arr, scale)

def preprocess_targets(targets, params, logger, scale=None):
    # calculate scaling
    if scale is None:
        mean  = np.nanmean(targets['train'], axis=0, keepdims=True)
        std   = np.nanstd (targets['train'], axis=0, keepdims=True)
        scale = {'shift': mean, 'mult': std}
    return _preprocess_array(targets, 'targets', params, logger, scale)

def postprocess_targets(targets_predict, scale):
    return _postprocess_array(targets_predict, scale)

def preprocess_features_noise(features_noise, params, logger, scale=None):
    return _preprocess_array(features_noise, 'features_noise', params, logger, scale)

def preprocess_targets_noise(targets_noise, params, logger, scale=None):
    # exit if nothing to do
    if targets_noise is None or (isinstance(targets_noise, dict) and dictarray_is_none(targets_noise)):
        return None
    # calculate scaling
    if scale is None:
        mean  = np.nanmean(targets_noise['train'], axis=0, keepdims=True)
        std   = np.nanstd (targets_noise['train'], axis=0, keepdims=True)
        scale = {'shift': mean, 'mult': std}
    return _preprocess_array(targets_noise, 'targets_noise', params, logger, scale)

def postprocess_targets_noise(targets_noise_predict, scale):
    return _postprocess_array(targets_noise_predict, scale)

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
    features_type = params['data']['features_type'].casefold()
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
    if 'TIME'.casefold()       == features_type or \
       'TIME_NOISE'.casefold() == features_type:
        raise NotImplementedError()
    elif 'RATE_DURATION'.casefold() == features_type:
        for i, key in enumerate(['RATE', 'DURATION']):
            cond_positions.append(_get_positions_from_histogram(
                    features[...,i],
                    fn_params['range'][key],
                    fn_params['n_bins'][key],
                    fn_params['relevant_bins_threshold'][key]
            ))
    else:
        raise ValueError(f"Unknown features_type: {features_type}")
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
    features_type = params['data']['features_type'].casefold()
    # extract conditional samples
    assert 1 < features.shape[0]
    if 'TIME'.casefold()       == features_type or \
       'TIME_NOISE'.casefold() == features_type:
        raise NotImplementedError()
    elif 'RATE_DURATION'.casefold() == features_type:
        threshold = [0.01, 0.10]
        features_cond, targets_cond = _filter_samples(features, targets, position, threshold)
    else:
        raise ValueError(f"Unknown features_type: {features_type}")
    return features_cond, targets_cond

###############################################################################

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

class FHN_Dataset(Dataset):
    def __init__(
            self,
            features,
            targets,
            features_noise=None,
            targets_noise=None,
            features_transform_fn=None,
            features_sub_length=None,
            features_sub_begin_random=False,
            noise_idx_random=True,
            item_return_order='yx'
    ):
        super().__init__()
        self.features = torch.from_numpy(features)
        self.targets  = torch.from_numpy(targets)
        assert self.features.size(0) == self.targets.size(0)
        if features_noise is not None:
            self.features_noise = torch.from_numpy(features_noise)
        else:
            self.features_noise = None
        if targets_noise is not None:
            self.targets_noise = torch.from_numpy(targets_noise)
        else:
            self.targets_noise = None
        self.features_transform_fn      = features_transform_fn
        self.features_sub_length        = features_sub_length
        self.features_sub_begin_random  = features_sub_begin_random
        self.noise_idx_random           = noise_idx_random
        self.item_return_order          = item_return_order.casefold()

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = self.features[idx]
        if self.features_noise is not None:
            if self.noise_idx_random:
                noise_idx = torch.randint(self.features_noise.size(0), (1,))[0]
            else:
                noise_idx = idx
            features_transformed = features + self.features_noise[noise_idx]
        else:
            noise_idx = None
            features_transformed = features
        if self.features_transform_fn is not None:
            features_transformed = self.features_transform_fn(features_transformed[None,...])[0]
        if self.features_sub_length and \
           self.features_sub_length < features.size(-1):
            if self.features_sub_begin_random:
                idx_begin = np.random.randint(features.size(-1) - self.features_sub_length)
            else:
                idx_begin = 0
            idx_end = idx_begin + self.features_sub_length
            features             = features            [...,idx_begin:idx_end]
            features_transformed = features_transformed[...,idx_begin:idx_end]

        targets = self.targets[idx]
        if self.targets_noise is not None:
            assert noise_idx is not None
            targets_noise = self.targets_noise[noise_idx]
            targets = torch.cat((targets, targets_noise), dim=0)

        if 'xx'.casefold() == self.item_return_order:
            sample = (targets, targets)
        elif 'xy'.casefold() == self.item_return_order:
            sample = (targets, features_transformed)
        elif 'yx'.casefold() == self.item_return_order:
            sample = (features_transformed, targets)
        elif 'yy'.casefold() == self.item_return_order:
            sample = (features_transformed, features)
        else:
            raise ValueError(f"Unknown item return order: {self.item_return_order}")
        return sample


def create_dataloader(params, logger, mode,
                      features, targets,
                      features_noise, targets_noise,
                      features_transform_fn=None,
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

    # set common arguments of dataset
    dataset_kwargs = {
        'features_transform_fn'    : features_transform_fn,
        'features_sub_length'      : params['data']['features_sub_length'],
        'features_sub_begin_random': params['data']['features_sub_begin_random'],
        'item_return_order'        : item_return_order
    }
    if dataloader_kwargs['shuffle']:
        dataset_kwargs['noise_idx_random'] = True
    else:
        dataset_kwargs['noise_idx_random'] = False

    # create the dataset
    logger.info('Create new dataset')
    if ModeKeys.TRAIN == mode:
        dataset = FHN_Dataset(features['train'],
                              targets['train'],
                              features_noise = features_noise['train'],
                              targets_noise  = targets_noise['train'],
                              **dataset_kwargs)
    elif ModeKeys.VALIDATE == mode:
        dataset = FHN_Dataset(features['validate'],
                              targets['validate'],
                              features_noise = features_noise['validate'],
                              targets_noise  = targets_noise['validate'],
                              **dataset_kwargs)
    else:
        dataset = FHN_Dataset(features['test'],
                              targets['test'],
                              features_noise = features_noise['test'],
                              targets_noise  = targets_noise['test'],
                              **dataset_kwargs)

    # create the dataloader
    logger.info('Create new dataloader')
    dataloader = DataLoader(dataset, batch_size=batch_size, **dataloader_kwargs)

    # output
    return dataloader

