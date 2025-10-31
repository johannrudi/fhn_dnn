"""
Handling of data.
"""

import inspect, logging, pathlib, os, sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from utils import Mode

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

def _load_memmap(data_file, cols_num, dtype=np.float32):
    data_points = cols_num
    data_file_size = os.path.getsize(data_file)
    data_rows = data_file_size // (data_points * 4)
    return np.memmap(
        data_file,
        dtype=dtype,
        mode='r',
        shape=(data_rows, data_points)
    )

def _load_array(data_file, cols_num=None, dtype=np.float32):
    try:
        # load assuming a regular numpy array
        return np.load(data_file)
    except ValueError:
        # instead, load using memmap
        assert cols_num is not None
        assert dtype is not None
        return _load_memmap(data_file, cols_num=cols_num, dtype=dtype)

def _load_and_split_arrays(data_params, logger=None):
    # set up logger
    if logger is None:
        logger = logging.getLogger(f"{__name__}.{inspect.currentframe().f_code.co_name}")

    # set options
    features_type = data_params['features_type'].casefold()
    targets_type  = data_params.get('targets_type', 'N/A').casefold()
    data_dir      = pathlib.Path(data_params['data_dir'])
    file_names = data_params.get(
            'file_names',
            {
                'features'           : 'fhn_Ntrain20000_state_Nt2000_dt0.2.npy',
                'features_test'      : 'fhn_Ntest2000_state_Nt2000_dt0.2.npy',
                'features_stats'     : 'fhn_Ntrain20000_state_stats.npy',
                'features_stats_test': 'fhn_Ntest2000_state_stats.npy',
                'targets'            : 'fhn_Ntrain20000_param.npy',
                'targets_test'       : 'fhn_Ntest2000_param.npy',
                'features_noise'     : 'ar1_Ntrain20000_state_Nt2000_dt0.2.npy',
                'features_noise_test': 'ar1_Ntest2000_state_Nt2000_dt0.2.npy',
                'targets_noise'      : 'ar1_Ntrain20000_param.npy',
                'targets_noise_test' : 'ar1_Ntest2000_param.npy'
            }
    )
    features_use_channels_range = data_params.get('features_use_channels_range', None)
    features_use_length_range   = data_params.get('features_use_length_range', None)
    Ntrain    = data_params.get('Ntrain')
    Nvalidate = data_params.get('Nvalidate')
    Ntest     = data_params.get('Ntest', None)

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
            logger.debug(f"{features_type=}, {features_.shape=}, {features_test.shape=}")
        elif features_type in ['ODE_STATS'.casefold(), 'RATE_DURATION'.casefold()]:
            rate          = np.load(data_dir/'fhn_T200_samplePrior_spikeRate.npy')
            duration      = np.load(data_dir/'fhn_T200_samplePrior_spikeDuration.npy')
            features_     = np.expand_dims( np.stack((rate, duration), axis=1), axis=1 )
            features_     = features_[:-Ntest,...]
            features_test = features_[-Ntest:,...]
            logger.debug(f"{features_type=}, {features_.shape=}, {features_test.shape=}")
        elif features_type == 'NOISE'.casefold():
            pass
        else:
            raise ValueError(f"Unknown {features_type=}")
        # load targets
        if targets_type in ['ODE'.casefold(), 'ODE_NOISE'.casefold()]:
            targets_     = np.load(data_dir/'fhn_T200_samplePrior_theta.npy')
            targets_     = targets_[:-Ntest,...]
            targets_test = targets_[-Ntest:,...]
            logger.debug(f"{targets_type=}, {targets_.shape=}, {targets_test.shape=}")
        elif targets_type in ['NOISE'.casefold(), 'N/A'.casefold()]:
            pass
        else:
            raise ValueError(f"Unknown {targets_type=}")
        # load features noise
        if features_type in ['TIME_NOISE'.casefold(), 'NOISE'.casefold()]:
            features_noise_     = np.expand_dims( np.load(data_dir/'noise_correlated_Nt1000_Nsim10000_data.npy'), axis=1 )
            features_noise_     = features_noise_[:-Ntest,...]
            features_noise_test = features_noise_[-Ntest:,...]
            logger.debug(f"{features_type=}, {features_noise_.shape=}, {features_noise_test.shape=}")
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
            logger.debug(f"{targets_type=}, {targets_noise_.shape=}, {targets_noise_test.shape=}")
        elif targets_type in ['ODE'.casefold(), 'N/A'.casefold()]:
            pass
        else:
            raise ValueError(f"Unknown {targets_type=}")
    elif '2025' in data_dir.name:
        # load features
        if features_type in ['TIME'.casefold(), 'TIME_NOISE'.casefold()]:
            features_ = _load_array(data_dir/file_names['features'])
            features_test = _load_array(data_dir/file_names['features_test'])
            # DEV data of AlonsoMarder model
            # features_ = _load_array(file_names['features'], cols_num=600000)[:,None,:] # TODO change the data dimensions, instead of this hack
            # features_test = _load_array(file_names['features_test'], cols_num=600000)[:,None,:] # TODO change the data dimensions, instead of this hack
            # /DEV
            if features_use_channels_range is not None:
                assert 2 == len(features_use_channels_range), len(features_use_channels_range)
                start, stop = features_use_channels_range
                features_     = features_[:,start:stop,:]
                features_test = features_test[:,start:stop,:]
            if features_use_length_range is not None:
                assert 2 == len(features_use_length_range), len(features_use_length_range)
                start, stop = features_use_length_range
                features_     = features_[:,:,start:stop]
                features_test = features_test[:,:,start:stop]
            if Ntest is None:
                Ntest = features_test.shape[0]
            logger.debug(f"{features_type=}, {features_.shape=}, {features_test.shape=}")
        elif features_type in ['ODE_STATS'.casefold(), 'RATE_DURATION'.casefold()]:
            features_     = np.expand_dims( _load_array(data_dir/file_names['features_stats']), axis=1 )
            features_test = np.expand_dims( _load_array(data_dir/file_names['features_stats_test']), axis=1 )
            # DEV data of AlonsoMarder model
            # features_ = _load_array(file_names['features_stats'], cols_num=2)[:,None,:] # TODO change the data dimensions, instead of this hack
            # features_test = _load_array(file_names['features_stats_test'], cols_num=2)[:,None,:] # TODO change the data dimensions, instead of this hack
            # /DEV
            if Ntest is None:
                Ntest = features_test.shape[0]
            logger.debug(f"{features_type=}, {features_.shape=}, {features_test.shape=}")
        elif features_type == 'NOISE'.casefold():
            pass
        else:
            raise ValueError(f"Unknown {features_type=}")
        # load targets
        if targets_type in ['ODE'.casefold(), 'ODE_NOISE'.casefold()]:
            targets_     = _load_array(data_dir/file_names['targets'])
            targets_test = _load_array(data_dir/file_names['targets_test'])
            # DEV data of AlonsoMarder model
            # targets_ = _load_array(file_names['targets'], cols_num=9) # TODO change the data dimensions, instead of this hack
            # targets_test = _load_array(file_names['targets_test'], cols_num=9) # TODO change the data dimensions, instead of this hack
            # /DEV
            logger.debug(f"{targets_type=}, {targets_.shape=}, {targets_test.shape=}")
        elif targets_type in ['NOISE'.casefold(), 'N/A'.casefold()]:
            pass
        else:
            raise ValueError(f"Unknown {targets_type=}")
        # load features of noise
        if features_type in ['NOISE'.casefold(), 'TIME_NOISE'.casefold()]:
            features_noise_     = np.load(data_dir/file_names['features_noise'])
            features_noise_test = np.load(data_dir/file_names['features_noise_test'])
            if features_use_channels_range is not None:
                assert 2 == len(features_use_channels_range), len(features_use_channels_range)
                start, stop = features_use_channels_range
                features_noise_     = features_noise_[:,start:stop,:]
                features_noise_test = features_noise_test[:,start:stop,:]
            if features_use_length_range is not None:
                assert 2 == len(features_use_length_range), len(features_use_length_range)
                start, stop = features_use_length_range
                features_noise_     = features_noise_[:,:,start:stop]
                features_noise_test = features_noise_test[:,:,start:stop]
            if Ntest is None:
                Ntest = features_noise_test.shape[0]
            logger.debug(f"{features_type=}, {features_noise_.shape=}, {features_noise_test.shape=}")
        elif features_type in ['TIME'.casefold(), 'ODE_STATS'.casefold(), 'RATE_DURATION'.casefold()]:
            pass
        else:
            raise ValueError(f"Unknown {features_type=}")
        # load targets of noise
        if targets_type in ['NOISE'.casefold(), 'ODE_NOISE'.casefold()]:
            targets_noise_     = np.load(data_dir/file_names['targets_noise'])
            targets_noise_test = np.load(data_dir/file_names['targets_noise_test'])
            logger.debug(f"{targets_type=}, {targets_noise_.shape=}, {targets_noise_test.shape=}")
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
    data_params = params['data']

    # read data and split files
    features, targets, features_noise, targets_noise = _load_and_split_arrays(data_params)

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
    if params['data'].get('features_sub_length') and \
       params['data']['features_sub_length'] < params['data']['num_features'][-1]:
        params['data']['num_features'][-1] = params['data']['features_sub_length']
    if params['data'].get('features_sub_step') and 1 < params['data']['features_sub_step']:
        params['data']['num_features'][-1] = (
            params['data']['num_features'][-1] // params['data']['features_sub_step']
        )

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

#def _log_transform(data, shift=0.0):
#    """ Applies log-transform for preprocessing. """
#    if isinstance(data, dict):
#        for key in data.keys():
#            data[key] = np.log(shift + data[key])
#    else:
#        data = np.log(shift + data)
#    return data

#def _log_transform_inverse(data, shift=0.0):
#    """ Applies inverse of log-transform for postprocessing. """
#    if isinstance(data, dict):
#        for key in data.keys():
#            data[key] = np.exp(data[key]) - shift
#    else:
#        data = np.exp(data) - shift
#    return data

def _apply_scale(data, scale):
    """ Applies scale for preprocessing. """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = (data[key] - scale['shift']) * (1.0/scale['mult'])
    else:
        data = (data - scale['shift']) * (1.0/scale['mult'])
    return data

def _apply_scale_inverse(data, scale):
    """ Applies inverse scale for postprocessing. """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = data[key] * scale['mult'] + scale['shift']
    else:
        data = data * scale['mult'] + scale['shift']
    return data

def preprocess_features(features, params, logger, scale=None, array_name='features'):
    # exit if nothing to do
    if dictarray_is_none(features):
        return None
    features_type = params['data']['features_type'].casefold()
# DEV
#   # apply transformation
#   if features_type == 'RATE_DURATION'.casefold():
#       for key in features.keys():
#           features[key][...,1] = _log_transform(features[key][...,1], shift=1.0)
#/DEV
    # calculate scaling for normalization
    if scale is None:
        scale = {
            'shift': 0.0,
            'mult':  1.0
        }
        if params['data'].get('features_normalize', False):
            if features_type in ['TIME'.casefold(), 'TIME_NOISE'.casefold(), 'NOISE'.casefold()]:
                assert 3 == features['train'].ndim
                scale = {
                    'shift': np.mean(features['train'], axis=(0,2), keepdims=True),
                    'mult' : np.std (features['train'], axis=(0,2), keepdims=True)
                }
            elif features_type in ['ODE_STATS'.casefold(), 'RATE_DURATION'.casefold()]:
                assert 3 == features['train'].ndim
                assert 1 == features['train'].shape[1]
                scale = {
                    'shift': np.nanmean(features['train'], axis=0, keepdims=True),
                    'mult' : np.nanstd (features['train'], axis=0, keepdims=True)
                }
# DEV
#               # override scaling of "spike rate"
#               if 'RATE_DURATION'.casefold() == features_type:
#                   features_min = np.nanmin(features['train'], axis=0, keepdims=True)
#                   features_max = np.nanmax(features['train'], axis=0, keepdims=True)
#                   scale['shift'][...,0] = features_min[...,0]
#                   scale['mult'][...,0]  = features_max[...,0] - features_min[...,0]
#/DEV
            else:
                raise NotImplementedError(f"Unknown {features_type=}")
    # apply scaling
    logger.info(f"{array_name} scale = {scale}")
    features = _apply_scale(features, scale)
    # replace nan values
    if features_type in ['TIME'.casefold(), 'TIME_NOISE'.casefold(), 'NOISE'.casefold()]:
        pass
    elif features_type in ['ODE_STATS'.casefold(), 'RATE_DURATION'.casefold()]:
        for key in features.keys():
            features[key] = np.where(np.isnan(features[key]), -10.0, features[key])
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
    features = _apply_scale_inverse(features, scale)
# DEV
#   # apply inverse scaling
#   if features_type == 'RATE_DURATION'.casefold():
#       features = _apply_scale_inverse(features, scale)
#       #TODO apply inverse transforms
#       raise NotImplementedError()
#/DEV

def preprocess_targets(targets, params, logger, scale=None, array_name='targets'):
    # exit if nothing to do
    if dictarray_is_none(targets):
        return None
    # calculate scaling for normalization
    if scale is None:
        scale = {
            'shift': 0.0,
            'mult':  1.0
        }
        if params['data'].get('targets_normalize', False):
            assert 2 == targets['train'].ndim
            scale = {
                'shift': np.mean(targets['train'], axis=0, keepdims=True),
                'mult' : np.std (targets['train'], axis=0, keepdims=True)
            }
    # apply scaling
    logger.info(f"{array_name} scale = {scale}")
    targets = _apply_scale(targets, scale)
    # return scale
    return scale

def postprocess_targets(targets, scale):
    # exit if nothing to do
    if dictarray_is_none(targets):
        return
    # apply inverse scaling
    targets = _apply_scale_inverse(targets, scale)

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
    """ Tuned for data set 2020-12-09. """
    data_dir      = params['data']['data_dir']
    features_type = params['data']['features_type'].casefold()
    # set function parameters
    fn_params = {
        'n_bins': {
            'TIME':     None,
            'RATE':     1000,
            'DURATION':   25,
        },
        'range': {
            'TIME':     None,
#           'RATE':     [ 0.55, 0.95],
#           'DURATION': [-0.50, 0.50],
            'RATE':     [-0.5, 1.0],
            'DURATION': [-0.5, 0.5],
        },
        'relevant_bins_threshold': {
            'TIME':     None,
            'RATE':     features.shape[0] * 0.06,
            'DURATION': features.shape[0] * 0.01,
        }
    }
    # extract conditional positions
    assert 1 < features.shape[0]
    cond_positions = list()
    if features_type in ['TIME'.casefold(), 'TIME_NOISE'.casefold(), 'NOISE'.casefold()]:
        # set indices of samples as positions for conditionals
        cond_positions.append(np.arange(features.shape[0] - 6, features.shape[0], dtype=np.int32))
    elif features_type == 'ODE_STATS'.casefold():
        raise NotImplementedError()
    elif features_type == 'RATE_DURATION'.casefold():
        if '2020' in data_dir:
#           cond_positions.append(np.array([ 0.60,  0.70,  0.80,  0.90]))
#           cond_positions.append(np.array([-0.36, -0.34, -0.26, -0.18, -0.17]))
            cond_positions.append(np.array([-0.4205, 0.1795, 0.7795]))
            cond_positions.append(np.array([-0.34, -0.3, -0.26, -0.22]))
        elif '2025' in data_dir:
            cond_positions.append(np.array([-0.419, 0.175, 0.769]))
            cond_positions.append(np.array([-0.5, -0.26, 0.22, 0.46]))
        else:  # otherwise find values from histogram
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
    # filter features
    if features.shape[-1] == len(position):
        # if as many positions as features: threshold positions across all features
        for i, (pos, thresh) in enumerate(zip(position, threshold)):
            features_ = features[...,i].flatten()
            idx_thresh = np.logical_and((pos - thresh) < features_, features_ < (pos + thresh))
            if 0 == i:
                indices = idx_thresh
            else:
                indices = np.logical_and(indices, idx_thresh)
    elif 1 == len(position) and features.shape[1:] == position[0].shape[1:]:
        # if positions are samples: threshold by the normed distance to samples
        sample = position[0]
        thresh = threshold[0]
        assert 1 == sample.shape[0]
        distances = np.linalg.norm(features - sample, axis=(1, 2))
        indices = np.where(distances < thresh)[0]
    features_filtered = features[indices]
    # apply filter to targets
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
    if features_type in ['TIME'.casefold(), 'TIME_NOISE'.casefold(), 'NOISE'.casefold()]:
        threshold = [0.01 * np.prod(features.shape[1:])]
        features_cond, targets_cond = _filter_samples(features, targets, position, threshold)
    elif features_type == 'ODE_STATS'.casefold():
        raise NotImplementedError()
    elif features_type == 'RATE_DURATION'.casefold():
        #threshold = [0.05, 0.15]
        threshold = [0.6, 0.6]
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
            features_sub_step=None,
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
        self.features_sub_step           = features_sub_step


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
        # truncate features with step length
        if self.features_sub_step and 1 < self.features_sub_step:
            features             = features            [...,::self.features_sub_step]
            features_transformed = features_transformed[...,::self.features_sub_step]
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
    if mode.any(Mode.TRAIN | Mode.PROFILE):
        shuffle = True
        batch_size = params['data']['train_batch_size']
    elif mode.any(Mode.VALIDATE | Mode.PREDICT | Mode.EVAL):
        shuffle = False
        batch_size = params['data']['eval_batch_size']
    else:
        raise NotImplementedError()

    # set arguments for dataset
    dataset_kwargs = dict(noise_idx_random=shuffle)

    # create the dataset
    logger.info('Create new dataset')
    dataset = FHN_Dataset(
        features, targets,
        features_noise = features_noise,
        targets_noise  = targets_noise,
        features_additive_noise_std = params['data'].get('features_additive_noise_std', 0.0),
        features_transform_fn       = features_transform_fn,
        features_sub_length         = params['data'].get('features_sub_length', 0),
        features_sub_begin_random   = params['data'].get('features_sub_begin_random', False),
        features_sub_step           = params['data'].get('features_sub_step'),
        item_return_order           = item_return_order,
        **dataset_kwargs
    )

    # set arguments for dataloader
    dataloader_kwargs = dict(
        shuffle = shuffle,
        drop_last = False,
        batch_size = batch_size,
    )
    if torch.cuda.is_available():
        n_workers = os.cpu_count() // 4
        in_order = not shuffle
        dataloader_kwargs.update(dict(
            num_workers=n_workers,   # CPU subprocesses for data loading
            pin_memory=True,         # faster CPU->GPU transfer
            prefetch_factor=2,       # batches to prefetch per worker
            persistent_workers=True, # keep workers alive between epochs
            multiprocessing_context='fork',  # (or 'spawn' on Windows)
            in_order=in_order,          # don't enforce first-in, first-out order
        ))
    else:  # otherwise CPU-only setup
        n_workers = min(os.cpu_count(), 2)
        in_order = not shuffle
        dataloader_kwargs.update(dict(
            num_workers=n_workers,
            prefetch_factor=2,
            persistent_workers=True,
            multiprocessing_context='fork',
            in_order=in_order,
        ))

    # create the dataloader
    logger.info('Create new dataloader')
    dataloader = DataLoader(dataset, **dataloader_kwargs)

    # output
    return dataloader
