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

def _load_and_split_arrays(data_dir, features_type, targets_type, Ntrain, Nvalidate, Ntest):
    # load arrays
    if '2020' in data_dir.name:
        # set default
        if Ntest is None:
            Ntest = 2000
        # load features
        if features_type in ['TIME'.casefold(), 'TIME_NOISE'.casefold()]:
            features_     = np.expand_dims( np.load(data_dir/'fhn_T200_samplePrior_state0.npy'), axis=1 )
            features_     = features_[:-Ntest,...]
            features_test = features_[-Ntest:,...]
        elif features_type in ['ODE_STATS'.casefold(), 'RATE_DURATION'.casefold()]:
            rate          = np.load(data_dir/'fhn_T200_samplePrior_spikeRate.npy')
            duration      = np.load(data_dir/'fhn_T200_samplePrior_spikeDuration.npy')
            features_     = np.expand_dims( np.stack((rate, duration), axis=1), axis=1 )
            features_     = features_[:-Ntest,...]
            features_test = features_[-Ntest:,...]
        elif features_type == 'NOISE'.casefold():
            pass
        else:
            raise ValueError(f"Unknown {features_type=}")
        # load targets
        if targets_type in ['ODE'.casefold(), 'ODE_NOISE'.casefold()]:
            targets_     = np.load(data_dir/'fhn_T200_samplePrior_theta.npy')
            targets_     = targets_[:-Ntest,...]
            targets_test = targets_[-Ntest:,...]
        elif targets_type in ['NOISE'.casefold(), 'N/A'.casefold()]:
            pass
        else:
            raise ValueError(f"Unknown {targets_type=}")
        # load features noise
        if features_type in ['TIME_NOISE'.casefold(), 'NOISE'.casefold()]:
            features_noise_     = np.expand_dims( np.load(data_dir/'noise_correlated_Nt1000_Nsim10000_data.npy'), axis=1 )
            features_noise_     = features_noise_[:-Ntest,...]
            features_noise_test = features_noise_[-Ntest:,...]
        elif features_type in ['TIME'.casefold(), 'ODE_STATS'.casefold(), 'RATE_DURATION'.casefold()]:
            pass
        else:
            raise ValueError(f"Unknown {features_type=}")
        # load targets of noise
        if targets_type in ['NOISE'.casefold(), 'ODE_NOISE'.casefold()]:
            noise_correl       = np.load(data_dir/'noise_correlated_Nt1000_Nsim10000_correlation.npy')
            noise_stddev       = np.load(data_dir/'noise_correlated_Nt1000_Nsim10000_stddev.npy')
            targets_noise_     = np.stack((noise_correl, noise_stddev), axis=1)
            targets_noise_     = targets_noise_[:-Ntest,...]
            targets_noise_test = targets_noise_[-Ntest:,...]
        elif targets_type in ['ODE'.casefold(), 'N/A'.casefold()]:
            pass
        else:
            raise ValueError(f"Unknown {targets_type=}")
    elif '2025' in data_dir.name:
        # load features
        if features_type in ['TIME'.casefold(), 'TIME_NOISE'.casefold()]:
            features_     = np.load(data_dir/'fhn_Ntrain20000_state_Nt2000_dt0.2.npy')[:,0:1,:]
            features_test = np.load(data_dir/'fhn_Ntest2000_state_Nt2000_dt0.2.npy')[:,0:1,:]
            if Ntest is None:
                Ntest = features_test.shape[0]
        elif features_type in ['ODE_STATS'.casefold(), 'RATE_DURATION'.casefold()]:
            features_     = np.expand_dims( np.load(data_dir/'fhn_Ntrain20000_state_stats.npy'), axis=1 )
            features_test = np.expand_dims( np.load(data_dir/'fhn_Ntest2000_state_stats.npy'), axis=1 )
            if Ntest is None:
                Ntest = features_test.shape[0]
        elif features_type == 'NOISE'.casefold():
            pass
        else:
            raise ValueError(f"Unknown {features_type=}")
        # load targets
        if targets_type in ['ODE'.casefold(), 'ODE_NOISE'.casefold()]:
            targets_     = np.load(data_dir/'fhn_Ntrain20000_param.npy')
            targets_test = np.load(data_dir/'fhn_Ntest2000_param.npy')
        elif targets_type in ['NOISE'.casefold(), 'N/A'.casefold()]:
            pass
        else:
            raise ValueError(f"Unknown {targets_type=}")
        # load features of noise
        if features_type in ['NOISE'.casefold(), 'TIME_NOISE'.casefold()]:
            features_noise_     = np.expand_dims( np.load(data_dir/'ar1_Ntrain20000_state_Nt2000_dt0.2.npy'), axis=1 )
            features_noise_test = np.expand_dims( np.load(data_dir/'ar1_Ntest2000_state_Nt2000_dt0.2.npy'), axis=1 )
            if Ntest is None:
                Ntest = features_noise_test.shape[0]
        elif features_type in ['TIME'.casefold(), 'ODE_STATS'.casefold(), 'RATE_DURATION'.casefold()]:
            pass
        else:
            raise ValueError(f"Unknown {features_type=}")
        # load targets of noise
        if targets_type in ['NOISE'.casefold(), 'ODE_NOISE'.casefold()]:
            targets_noise_     = np.load(data_dir/'ar1_Ntrain20000_param.npy')
            targets_noise_test = np.load(data_dir/'ar1_Ntest2000_param.npy')
        elif targets_type in ['ODE'.casefold(), 'N/A'.casefold()]:
            pass
        else:
            raise ValueError(f"Unknown {targets_type=}")
    else:
        raise NotImplementedError(f"Unsupported data directory: {data_dir}")

    # split arrays
    if features_type in ['TIME'.casefold(), 'TIME_NOISE'.casefold(), 'ODE_STATS'.casefold(), 'RATE_DURATION'.casefold()]:
        ft_train    = features_[:Ntrain,...]     if 0 < Ntrain    else np.array([])
        ft_validate = features_[-Nvalidate:,...] if 0 < Nvalidate else np.array([])
        ft_test     = features_test[:Ntest,...]  if 0 < Ntest     else np.array([])
        features = dictarray_set(ft_train, ft_validate, ft_test)
    else:
        features = dictarray_empty()
    if targets_type in ['ODE'.casefold(), 'ODE_NOISE'.casefold()]:
        tg_train    = targets_[:Ntrain,...]     if 0 < Ntrain    else np.array([])
        tg_validate = targets_[-Nvalidate:,...] if 0 < Nvalidate else np.array([])
        tg_test     = targets_test[:Ntest,...]  if 0 < Ntest     else np.array([])
        targets = dictarray_set(tg_train, tg_validate, tg_test)
    else:
        targets = dictarray_empty()
    if features_type in ['NOISE'.casefold(), 'TIME_NOISE'.casefold()]:
        ft_train    = features_noise_[:Ntrain,...]     if 0 < Ntrain    else np.array([])
        ft_validate = features_noise_[-Nvalidate:,...] if 0 < Nvalidate else np.array([])
        ft_test     = features_noise_test[:Ntest,...]  if 0 < Ntest     else np.array([])
        features_noise = dictarray_set(ft_train, ft_validate, ft_test)
    else:
        features_noise = dictarray_empty()
    if targets_type in ['NOISE'.casefold(), 'ODE_NOISE'.casefold()]:
        tg_train    = targets_noise_[:Ntrain,...]     if 0 < Ntrain    else np.array([])
        tg_validate = targets_noise_[-Nvalidate:,...] if 0 < Nvalidate else np.array([])
        tg_test     = targets_noise_test[:Ntest,...]  if 0 < Ntest     else np.array([])
        targets_noise = dictarray_set(tg_train, tg_validate, tg_test)
    else:
        targets_noise  = dictarray_empty()

    # return features and targets
    return features, targets, features_noise, targets_noise

def load_data(params, logger):
    data_params   = params['data']
    data_dir      = pathlib.Path(data_params['data_dir'])
    features_type = data_params['features_type'].casefold()
    targets_type  = data_params.get('targets_type', 'N/A').casefold()

    # read data and split files
    features, targets, features_noise, targets_noise = _load_and_split_arrays(
            data_dir,
            features_type,
            targets_type,
            data_params.get('Ntrain'),
            data_params.get('Nvalidate'),
            data_params.get('Ntest', None)
    )

    # print info
    if dictarray_is_not_none(features):
        for key in features.keys():
            logger.info(f"features['{key}']:\tshape {features[key].shape}, dtype {features[key].dtype}")
    if dictarray_is_not_none(targets):
        for key in targets.keys():
            logger.info(f"targets['{key}']: \tshape {targets[key].shape}, dtype {targets[key].dtype}")
    if dictarray_is_not_none(features_noise):
        for key in features_noise.keys():
            logger.info(f"features_noise['{key}']:\tshape {features_noise[key].shape}, dtype {features_noise[key].dtype}")
    if dictarray_is_not_none(targets_noise):
        for key in targets_noise.keys():
            logger.info(f"targets_noise['{key}']: \tshape {targets_noise[key].shape}, dtype {targets_noise[key].dtype}")

    # set feature sizes
    if dictarray_is_not_none(features):
        params['data']['num_features'] = list(features['train'].shape[1:])
        params['data'].setdefault('Ntest', features['test'].shape[0])
    elif dictarray_is_not_none(features_noise):
        params['data']['num_features'] = list(features_noise['train'].shape[1:])
        params['data'].setdefault('Ntest', features_noise['test'].shape[0])
    else:
        raise NotImplementedError()
    # set reduced feature sizes
    if params['data']['features_sub_length'] and \
       params['data']['features_sub_length'] < params['data']['num_features'][-1]:
        params['data']['num_features'][-1] = params['data']['features_sub_length']

    # set targets sizes
    params['data']['num_targets'] = 0
    if dictarray_is_not_none(targets):
        assert dictarray_is_not_none(features)
        params['data']['num_targets'] += targets['train'].shape[1]
    if dictarray_is_not_none(targets_noise):
        assert dictarray_is_not_none(features_noise)
        params['data']['num_targets'] += targets_noise['train'].shape[1]

    # print sample sizes
    logger.info(f"Ntrain:    {data_params['Ntrain']}")
    logger.info(f"Nvalidate: {data_params['Nvalidate']}")
    logger.info(f"Ntest:     {data_params['Ntest']}")

    # print data shapes
    logger.debug(f"num_features: {params['data']['num_features']}")
    logger.debug(f"num_targets:  {params['data']['num_targets']}")

    # return data
    return features, targets, features_noise, targets_noise

def load_timesteps(params):
    data_params = params['data']
    data_dir    = pathlib.Path(data_params['data_dir'])
    if '2020' in data_dir.name:
        timesteps = np.load(data_dir/'fhn_T200_samplePrior_time.npy')
    elif '2025' in data_dir.name:
        timesteps = np.load(data_dir/'fhn_timesteps_Nt2000_dt0.2.npy')
    else:
        raise NotImplementedError(f"Unsupported data directory: {data_dir}")
    return timesteps

###############################################################################

def _log_transform(data, shift=0.0):
    """ Applies log-transform for preprocessing. """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = np.log(shift + data[key])
    else:
        data = np.log(shift + data)

def _log_transform_inverse(data, shift=0.0):
    """ Applies inverse of log-transform for postprocessing. """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = np.exp(data[key]) - shift
    else:
        data = np.exp(data) - shift

def _apply_scale(data, scale):
    """ Applies scale for preprocessing. """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = (data[key] - scale['shift']) * (1.0/scale['mult'])
    else:
        data = (data - scale['shift']) * (1.0/scale['mult'])

def _apply_scale_inverse(data, scale):
    """ Applies inverse scale for postprocessing. """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = data[key] * scale['mult'] + scale['shift']
    else:
        data = data * scale['mult'] + scale['shift']

def preprocess_features(features, params, logger, scale=None, array_name='features'):
    # exit if nothing to do
    if dictarray_is_none(features):
        return None
    features_type = params['data']['features_type'].casefold()
    # apply transformation
    if features_type in ['TIME'.casefold(), 'TIME_NOISE'.casefold(), 'NOISE'.casefold(), 'ODE_STATS'.casefold()]:
        pass
    elif features_type == 'RATE_DURATION'.casefold():
        for key in features.keys():
            _log_transform(features[key][...,1], shift=1.0)
    else:
        raise NotImplementedError(f"Unknown {features_type=}")
    # calculate scaling
    if scale is None:
        if features_type in ['TIME'.casefold(), 'TIME_NOISE'.casefold(), 'NOISE'.casefold()]:
            scale = {'shift': 0.0, 'mult': 1.0}
        elif features_type in ['ODE_STATS'.casefold(), 'RATE_DURATION'.casefold()]:
            assert 3 == features['train'].ndim
            assert 1 == features['train'].shape[1]
            scale = {
                'shift': np.nanmean(features['train'], axis=0, keepdims=True),
                'mult' : np.nanstd (features['train'], axis=0, keepdims=True)
            }
            # override scaling of "spike rate"
            if 'RATE_DURATION'.casefold() == features_type:
                features_min = np.nanmin(features['train'], axis=0, keepdims=True)
                features_max = np.nanmax(features['train'], axis=0, keepdims=True)
                scale['shift'][...,0] = features_min[...,0]
                scale['mult'][...,0]  = features_max[...,0] - features_min[...,0]
        else:
            raise NotImplementedError(f"Unknown {features_type=}")
    # apply scaling
    logger.info(f"{array_name} scale = {scale}")
    _apply_scale(features, scale)
    # replace nan values
    if features_type in ['TIME'.casefold(), 'TIME_NOISE'.casefold(), 'NOISE'.casefold()]:
       pass
    elif features_type in ['ODE_STATS'.casefold(), 'RATE_DURATION'.casefold()]:
        for key in features.keys():
            features[key] = np.where(np.isnan(features[key]), -1.0, features[key])
    else:
        raise NotImplementedError(f"Unknown {features_type=}")
    # return scale
    return scale

def postprocess_features(features, scale, params):
    # exit if nothing to do
    if dictarray_is_none(features):
        return
    features_type = params['data']['features_type'].casefold()
    # apply inverse scaling
    if features_type in ['TIME'.casefold(), 'TIME_NOISE'.casefold(), 'NOISE'.casefold(), 'ODE_STATS'.casefold()]:
        _apply_scale_inverse(features, scale)
    elif features_type == 'RATE_DURATION'.casefold():
        _apply_scale_inverse(features, scale)
        #TODO apply inverse transforms
        raise NotImplementedError()
    else:
        raise NotImplementedError(f"Unknown {features_type=}")

def preprocess_targets(targets, params, logger, scale=None, array_name='targets'):
    # exit if nothing to do
    if dictarray_is_none(targets):
        return None
    # calculate scaling
    if scale is None:
        scale = {
            'shift': np.mean(targets['train'], axis=0, keepdims=True),
            'mult' : np.std (targets['train'], axis=0, keepdims=True)
        }
    # apply scaling
    logger.info(f"{array_name} scale = {scale}")
    _apply_scale(targets, scale)
    # return scale
    return scale

def postprocess_targets(targets, scale):
    # exit if nothing to do
    if dictarray_is_none(targets):
        return None
    # apply inverse scaling
    _apply_scale_inverse(targets, scale)

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
    if features_type in ['TIME'.casefold(), 'TIME_NOISE'.casefold(), 'NOISE'.casefold(), 'ODE_STATS'.casefold()]:
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
    if features_type in ['TIME'.casefold(), 'TIME_NOISE'.casefold(), 'NOISE'.casefold(), 'ODE_STATS'.casefold()]:
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
            features_additive_noise_std=0.0,
            features_transform_fn=None,
            features_sub_length=None,
            features_sub_begin_random=False,
            noise_idx_random=True,
            item_return_order='yx'
    ):
        super().__init__()
        assert features is not None or features_noise is not None
        assert targets is None or features is not None
        assert targets_noise is None or features_noise is not None
        # set arrays from arguments
        if features is not None:
            self.features = torch.from_numpy(features)
        else:
            self.features = None
        if targets is not None:
            self.targets = torch.from_numpy(targets)
            assert self.targets.size(0) == self.features.size(0)
        else:
            self.targets = None
        if features_noise is not None:
            self.features_noise = torch.from_numpy(features_noise)
        else:
            self.features_noise = None
        if targets_noise is not None:
            self.targets_noise = torch.from_numpy(targets_noise)
        else:
            self.targets_noise = None
        # set from arguments
        self.features_additive_noise_std = features_additive_noise_std
        self.features_transform_fn       = features_transform_fn
        self.features_sub_length         = features_sub_length
        self.features_sub_begin_random   = features_sub_begin_random
        self.noise_idx_random            = noise_idx_random
        self.item_return_order           = item_return_order.casefold()

    def __len__(self):
        if self.features is not None:
            return self.features.size(0)
        else:
            return self.features_noise.size(0)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # get feature sample
        if self.features is not None:
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
        elif self.features_noise is not None:
            features = self.features_noise[idx]
            features_transformed = features
        else:
            features = None
            features_transformed = None
        # apply additive i.i.d. noise
        if self.features_additive_noise_std:
            features_transformed += self.features_additive_noise_std * torch.randn(features_transformed.size())
        # transform features
        if self.features_transform_fn is not None:
            features_transformed = self.features_transform_fn(features_transformed[None,...])[0]
        # truncate features array
        if self.features_sub_length and \
           self.features_sub_length < features.size(-1):
            if self.features_sub_begin_random:
                idx_begin = np.random.randint(features.size(-1) - self.features_sub_length)
            else:
                idx_begin = 0
            idx_end = idx_begin + self.features_sub_length
            features             = features            [...,idx_begin:idx_end]
            features_transformed = features_transformed[...,idx_begin:idx_end]
        # get target sample
        if self.targets is not None:
            targets = self.targets[idx]
            if self.targets_noise is not None:
                assert noise_idx is not None
                targets_noise = self.targets_noise[noise_idx]
                targets = torch.cat((targets, targets_noise), dim=0)
        elif self.targets_noise is not None:
            targets = self.targets_noise[idx]
        else:
            targets = None
        # return sample
        if 'xx'.casefold() == self.item_return_order:
            assert targets is not None
            return (targets, targets)
        elif 'xy'.casefold() == self.item_return_order:
            assert features_transformed is not None and targets is not None
            return (targets, features_transformed)
        elif 'yx'.casefold() == self.item_return_order:
            assert features_transformed is not None and targets is not None
            return (features_transformed, targets)
        elif 'yy'.casefold() == self.item_return_order:
            assert features_transformed is not None and features is not None
            return (features_transformed, features)
        else:
            raise ValueError(f"Unknown item return order: {self.item_return_order}")


def create_dataloader(params, logger, mode,
                      features, targets,
                      features_noise, targets_noise,
                      features_transform_fn=None,
                      item_return_order='yx'):
    """ Creates a PyTorch dataset and dataloader from numpy arrays.
        Ref: https://pytorch.org/docs/stable/data.html
    """
    if ModeKeys.TRAIN == mode:
        dataset_kwargs    = {'noise_idx_random': True}
        dataloader_kwargs = {'shuffle': True, 'drop_last': False,
                             'batch_size': params['data']['train_batch_size']}
    elif ModeKeys.VALIDATE == mode or ModeKeys.EVAL == mode:
        dataset_kwargs    = {'noise_idx_random': False}
        dataloader_kwargs = {'shuffle': False, 'drop_last': False,
                             'batch_size': params['data']['eval_batch_size']}
    else:
        raise NotImplementedError()

    # create the dataset
    logger.info('Create new dataset')
    dataset = FHN_Dataset(
            features, targets,
            features_noise = features_noise,
            targets_noise  = targets_noise,
            features_additive_noise_std = params['data']['features_additive_noise_std'],
            features_transform_fn       = features_transform_fn,
            features_sub_length         = params['data']['features_sub_length'],
            features_sub_begin_random   = params['data']['features_sub_begin_random'],
            item_return_order           = item_return_order,
            **dataset_kwargs
    )

    # create the dataloader
    logger.info('Create new dataloader')
    dataloader = DataLoader(dataset, **dataloader_kwargs)

    # output
    return dataloader
