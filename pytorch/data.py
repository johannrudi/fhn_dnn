"""
Handling of data.
"""

import inspect
import logging
import os  # TODO: use pathlib instead
import pathlib

import numpy as np
import torch
from dlk.data.loader import DataLoaderConfig
from dlk.mode import Mode
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------


def dictarray_empty():
    return {"train": None, "validate": None, "test": None}


def dictarray_set(arr_train, arr_validate, arr_test):
    return {"train": arr_train, "validate": arr_validate, "test": arr_test}


def dictarray_is_none(arr):
    return (
        arr is None
        or arr["train"] is None
        or arr["validate"] is None
        or arr["test"] is None
    )


def dictarray_is_not_none(arr):
    return not dictarray_is_none(arr)


# ---------------------------------------


def _resolve_data_params(params, scope):
    """Merge the shared `data` block with the train/evaluate-specific blocks.

    Args:
        params (dict): full parameter tree with `data`, `data_train`, `data_evaluate`.
        scope (str): one of "train", "evaluate", "all". "all" merges both
            mode-specific blocks on top of `data`, with `data_train` winning
            over `data_evaluate` on any conflicting key.

    Returns:
        dict: flat dict combining the relevant blocks.

    Note:
        On conflicting duplicated keys under scope "all", `data_train` wins
        while mode-specific dataloaders still follow their own block.
    """
    if scope == "train":
        return {**params["data"], **params["data_train"]}
    elif scope == "evaluate":
        return {**params["data"], **params["data_evaluate"]}
    elif scope == "all":
        return {**params["data"], **params["data_evaluate"], **params["data_train"]}
    else:
        raise ValueError(f"Unknown scope: {scope}")


def _load_memmap(data_file, cols_num, dtype=np.float32):
    data_points = cols_num
    data_file_size = os.path.getsize(data_file)  # TODO: use pathlib instead
    data_rows = data_file_size // (data_points * 4)
    return np.memmap(data_file, dtype=dtype, mode="r", shape=(data_rows, data_points))


def _load_array(data_file, cols_num=None, dtype=np.float32, expand_dims_axis=None):
    # load numpy file
    try:
        # load assuming a regular numpy array
        array = np.load(data_file)
    except ValueError:
        # instead, load using memmap
        assert cols_num is not None
        assert dtype is not None
        array = _load_memmap(data_file, cols_num=cols_num, dtype=dtype)
    assert 2 <= array.ndim
    # ensure a channel dimension exists
    if expand_dims_axis is not None and array.ndim < 3:
        array = np.expand_dims(array, axis=expand_dims_axis)
    return array


def _load_and_split_arrays(data_params, logger=None):
    # set up logger
    if logger is None:
        frame = inspect.currentframe()
        logger = logging.getLogger(
            f"{__name__}.{frame.f_code.co_name}" if frame is not None else __name__
        )

    # set options
    features_type = data_params["features_type"].casefold()
    targets_type = data_params.get("targets_type", "N/A").casefold()
    data_dir = pathlib.Path(data_params["data_dir"])
    file_names = data_params.get(
        "file_names",
        {
            "features": "fhn_Ntrain20000_state_Nt2000_dt0.2.npy",
            "features_test": "fhn_Ntest2000_state_Nt2000_dt0.2.npy",
            "features_stats": "fhn_Ntrain20000_state_stats.npy",
            "features_stats_test": "fhn_Ntest2000_state_stats.npy",
            "targets": "fhn_Ntrain20000_param.npy",
            "targets_test": "fhn_Ntest2000_param.npy",
            "features_noise": "ar1_Ntrain20000_state_Nt2000_dt0.2.npy",
            "features_noise_test": "ar1_Ntest2000_state_Nt2000_dt0.2.npy",
            "targets_noise": "ar1_Ntrain20000_param.npy",
            "targets_noise_test": "ar1_Ntest2000_param.npy",
        },
    )
    features_use_channels_range = data_params.get("features_use_channels_range")
    features_use_length_range = data_params.get("features_use_length_range")
    features_cols_num = data_params.get("features_cols_num")  # for memmap
    targets_cols_num = data_params.get("targets_cols_num")  # for memmap
    features_noise_cols_num = data_params.get("features_noise_cols_num")  # for memmap
    targets_noise_cols_num = data_params.get("targets_noise_cols_num")  # for memmap
    Ntrain = data_params["Ntrain"]
    Nvalidate = data_params.get("Nvalidate")
    Ntest = data_params.get("Ntest")

    # load arrays
    if "2020" in data_dir.name:
        # set default
        if Ntest is None:
            Ntest = 2000
        # load features
        if features_type in ["TIME".casefold(), "TIME_NOISE".casefold()]:
            features_ = np.expand_dims(
                np.load(data_dir / "fhn_T200_samplePrior_state0.npy"), axis=1
            )
            features_ = features_[:-Ntest, ...]
            features_test = features_[-Ntest:, ...]
            logger.debug(
                f"{features_type=}, {features_.shape=}, {features_test.shape=}"
            )
        elif features_type in ["ODE_STATS".casefold(), "RATE_DURATION".casefold()]:
            rate = np.load(data_dir / "fhn_T200_samplePrior_spikeRate.npy")
            duration = np.load(data_dir / "fhn_T200_samplePrior_spikeDuration.npy")
            features_ = np.expand_dims(np.stack((rate, duration), axis=1), axis=1)
            features_ = features_[:-Ntest, ...]
            features_test = features_[-Ntest:, ...]
            logger.debug(
                f"{features_type=}, {features_.shape=}, {features_test.shape=}"
            )
        elif features_type == "NOISE".casefold():
            pass
        else:
            raise ValueError(f"Unknown {features_type=}")
        # load targets
        if targets_type in ["ODE".casefold(), "ODE_NOISE".casefold()]:
            targets_ = np.load(data_dir / "fhn_T200_samplePrior_theta.npy")
            targets_ = targets_[:-Ntest, ...]
            targets_test = targets_[-Ntest:, ...]
            logger.debug(f"{targets_type=}, {targets_.shape=}, {targets_test.shape=}")
        elif targets_type in ["NOISE".casefold(), "N/A".casefold()]:
            pass
        else:
            raise ValueError(f"Unknown {targets_type=}")
        # load features noise
        if features_type in ["TIME_NOISE".casefold(), "NOISE".casefold()]:
            features_noise_ = np.expand_dims(
                np.load(data_dir / "noise_correlated_Nt1000_Nsim10000_data.npy"), axis=1
            )
            features_noise_ = features_noise_[:-Ntest, ...]
            features_noise_test = features_noise_[-Ntest:, ...]
            logger.debug(
                f"{features_type=}, {features_noise_.shape=}, {features_noise_test.shape=}"
            )
        elif features_type in [
            "TIME".casefold(),
            "ODE_STATS".casefold(),
            "RATE_DURATION".casefold(),
        ]:
            pass
        else:
            raise ValueError(f"Unknown {features_type=}")
        # load targets of noise
        if targets_type in ["NOISE".casefold(), "ODE_NOISE".casefold()]:
            noise_correl = np.load(
                data_dir / "noise_correlated_Nt1000_Nsim10000_correlation.npy"
            )
            noise_stddev = np.load(
                data_dir / "noise_correlated_Nt1000_Nsim10000_stddev.npy"
            )
            targets_noise_ = np.stack((noise_correl, noise_stddev), axis=1)
            targets_noise_ = targets_noise_[:-Ntest, ...]
            targets_noise_test = targets_noise_[-Ntest:, ...]
            logger.debug(
                f"{targets_type=}, {targets_noise_.shape=}, {targets_noise_test.shape=}"
            )
        elif targets_type in ["ODE".casefold(), "N/A".casefold()]:
            pass
        else:
            raise ValueError(f"Unknown {targets_type=}")
    else:
        # load features
        if features_type in ["TIME".casefold(), "TIME_NOISE".casefold()]:
            features_ = _load_array(
                data_dir / file_names["features"],
                cols_num=features_cols_num,
                expand_dims_axis=1,
            )
            features_test = _load_array(
                data_dir / file_names["features_test"],
                cols_num=features_cols_num,
                expand_dims_axis=1,
            )
            if features_use_channels_range is not None:
                assert 2 == len(features_use_channels_range), len(
                    features_use_channels_range
                )
                start, stop = features_use_channels_range
                features_ = features_[:, start:stop, :]
                features_test = features_test[:, start:stop, :]
            if features_use_length_range is not None:
                assert 2 == len(features_use_length_range), len(
                    features_use_length_range
                )
                start, stop = features_use_length_range
                features_ = features_[:, :, start:stop]
                features_test = features_test[:, :, start:stop]
            if Ntest is None:
                Ntest = features_test.shape[0]
            logger.debug(
                f"{features_type=}, {features_.shape=}, {features_test.shape=}"
            )
        elif features_type in ["ODE_STATS".casefold(), "RATE_DURATION".casefold()]:
            features_ = _load_array(
                data_dir / file_names["features_stats"],
                cols_num=features_cols_num,
                expand_dims_axis=1,
            )
            features_test = _load_array(
                data_dir / file_names["features_stats_test"],
                cols_num=features_cols_num,
                expand_dims_axis=1,
            )
            if Ntest is None:
                Ntest = features_test.shape[0]
            logger.debug(
                f"{features_type=}, {features_.shape=}, {features_test.shape=}"
            )
        elif features_type == "NOISE".casefold():
            pass
        else:
            raise ValueError(f"Unknown {features_type=}")
        # load targets
        if targets_type in ["ODE".casefold(), "ODE_NOISE".casefold()]:
            targets_ = _load_array(
                data_dir / file_names["targets"],
                cols_num=targets_cols_num,
            )
            targets_test = _load_array(
                data_dir / file_names["targets_test"],
                cols_num=targets_cols_num,
            )
            logger.debug(f"{targets_type=}, {targets_.shape=}, {targets_test.shape=}")
        elif targets_type in ["NOISE".casefold(), "N/A".casefold()]:
            pass
        else:
            raise ValueError(f"Unknown {targets_type=}")
        # load features of noise
        if features_type in ["NOISE".casefold(), "TIME_NOISE".casefold()]:
            features_noise_ = _load_array(
                data_dir / file_names["features_noise"],
                cols_num=features_noise_cols_num,
                expand_dims_axis=1,
            )
            features_noise_test = _load_array(
                data_dir / file_names["features_noise_test"],
                cols_num=features_noise_cols_num,
                expand_dims_axis=1,
            )
            if features_use_channels_range is not None:
                assert 2 == len(features_use_channels_range), len(
                    features_use_channels_range
                )
                start, stop = features_use_channels_range
                features_noise_ = features_noise_[:, start:stop, :]
                features_noise_test = features_noise_test[:, start:stop, :]
            if features_use_length_range is not None:
                assert 2 == len(features_use_length_range), len(
                    features_use_length_range
                )
                start, stop = features_use_length_range
                features_noise_ = features_noise_[:, :, start:stop]
                features_noise_test = features_noise_test[:, :, start:stop]
            if Ntest is None:
                Ntest = features_noise_test.shape[0]
            logger.debug(
                f"{features_type=}, {features_noise_.shape=}, {features_noise_test.shape=}"
            )
        elif features_type in [
            "TIME".casefold(),
            "ODE_STATS".casefold(),
            "RATE_DURATION".casefold(),
        ]:
            pass
        else:
            raise ValueError(f"Unknown {features_type=}")
        # load targets of noise
        if targets_type in ["NOISE".casefold(), "ODE_NOISE".casefold()]:
            targets_noise_ = _load_array(
                data_dir / file_names["targets_noise"],
                cols_num=targets_noise_cols_num,
            )
            targets_noise_test = _load_array(
                data_dir / file_names["targets_noise_test"],
                cols_num=targets_noise_cols_num,
            )
            logger.debug(
                f"{targets_type=}, {targets_noise_.shape=}, {targets_noise_test.shape=}"
            )
        elif targets_type in ["ODE".casefold(), "N/A".casefold()]:
            pass
        else:
            raise ValueError(f"Unknown {targets_type=}")

    # split arrays
    if features_type in [
        "TIME".casefold(),
        "TIME_NOISE".casefold(),
        "ODE_STATS".casefold(),
        "RATE_DURATION".casefold(),
    ]:
        ft_train = features_[:Ntrain, ...] if 0 < Ntrain else np.array([])
        ft_validate = features_[-Nvalidate:, ...] if 0 < Nvalidate else np.array([])
        ft_test = features_test[:Ntest, ...] if 0 < Ntest else np.array([])
        features = dictarray_set(ft_train, ft_validate, ft_test)
    else:
        features = dictarray_empty()
    if targets_type in ["ODE".casefold(), "ODE_NOISE".casefold()]:
        tg_train = targets_[:Ntrain, ...] if 0 < Ntrain else np.array([])
        tg_validate = targets_[-Nvalidate:, ...] if 0 < Nvalidate else np.array([])
        tg_test = targets_test[:Ntest, ...] if 0 < Ntest else np.array([])
        targets = dictarray_set(tg_train, tg_validate, tg_test)
    else:
        targets = dictarray_empty()
    if features_type in ["NOISE".casefold(), "TIME_NOISE".casefold()]:
        ft_train = features_noise_[:Ntrain, ...] if 0 < Ntrain else np.array([])
        ft_validate = (
            features_noise_[-Nvalidate:, ...] if 0 < Nvalidate else np.array([])
        )
        ft_test = features_noise_test[:Ntest, ...] if 0 < Ntest else np.array([])
        features_noise = dictarray_set(ft_train, ft_validate, ft_test)
    else:
        features_noise = dictarray_empty()
    if targets_type in ["NOISE".casefold(), "ODE_NOISE".casefold()]:
        tg_train = targets_noise_[:Ntrain, ...] if 0 < Ntrain else np.array([])
        tg_validate = (
            targets_noise_[-Nvalidate:, ...] if 0 < Nvalidate else np.array([])
        )
        tg_test = targets_noise_test[:Ntest, ...] if 0 < Ntest else np.array([])
        targets_noise = dictarray_set(tg_train, tg_validate, tg_test)
    else:
        targets_noise = dictarray_empty()

    # return features and targets
    return features, targets, features_noise, targets_noise


def load_data(params, logger):
    data_params = _resolve_data_params(params, "all")

    # read data and split files
    features, targets, features_noise, targets_noise = _load_and_split_arrays(
        data_params
    )

    # print info
    if dictarray_is_not_none(features):
        for key in features.keys():
            logger.info(
                f"features['{key}']:\tshape {features[key].shape}, dtype {features[key].dtype}"
            )
    if dictarray_is_not_none(targets):
        for key in targets.keys():
            logger.info(
                f"targets['{key}']: \tshape {targets[key].shape}, dtype {targets[key].dtype}"
            )
    if dictarray_is_not_none(features_noise):
        for key in features_noise.keys():
            logger.info(
                f"features_noise['{key}']:\tshape {features_noise[key].shape}, dtype {features_noise[key].dtype}"
            )
    if dictarray_is_not_none(targets_noise):
        for key in targets_noise.keys():
            logger.info(
                f"targets_noise['{key}']: \tshape {targets_noise[key].shape}, dtype {targets_noise[key].dtype}"
            )

    # set feature sizes
    if "num_features" not in params["data"]:
        if dictarray_is_not_none(features):
            params["data"]["num_features"] = list(features["train"].shape[1:])
            params["data_evaluate"].setdefault("Ntest", features["test"].shape[0])
        elif dictarray_is_not_none(features_noise):
            params["data"]["num_features"] = list(features_noise["train"].shape[1:])
            params["data_evaluate"].setdefault("Ntest", features_noise["test"].shape[0])
        else:
            raise NotImplementedError()
        # set reduced feature sizes
        if (
            data_params.get("features_sub_length")
            and data_params["features_sub_length"] < params["data"]["num_features"][-1]
        ):
            params["data"]["num_features"][-1] = data_params["features_sub_length"]
        if (
            data_params.get("features_sub_step")
            and 1 < data_params["features_sub_step"]
        ):
            params["data"]["num_features"][-1] = (
                params["data"]["num_features"][-1] // data_params["features_sub_step"]
            )

    # set targets sizes
    if "num_targets" not in params["data"]:
        num_targets = 0
        if dictarray_is_not_none(targets):
            assert dictarray_is_not_none(features)
            num_targets += targets["train"].shape[1]
        if dictarray_is_not_none(targets_noise):
            assert dictarray_is_not_none(features_noise)
            num_targets += targets_noise["train"].shape[1]
        params["data"]["num_targets"] = [num_targets]

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
    data_params = params["data"]
    data_dir = pathlib.Path(data_params["data_dir"])
    if "2020" in data_dir.name:
        timesteps = np.load(data_dir / "fhn_T200_samplePrior_time.npy")
    elif "2025" in data_dir.name:
        timesteps = np.load(data_dir / "fhn_timesteps_Nt2000_dt0.2.npy")
    else:
        raise NotImplementedError(f"Unsupported data directory: {data_dir}")
    return timesteps


# ---------------------------------------

# def _log_transform(data, shift=0.0):
#    """ Applies log-transform for preprocessing. """
#    if isinstance(data, dict):
#        for key in data.keys():
#            data[key] = np.log(shift + data[key])
#    else:
#        data = np.log(shift + data)
#    return data

# def _log_transform_inverse(data, shift=0.0):
#    """ Applies inverse of log-transform for postprocessing. """
#    if isinstance(data, dict):
#        for key in data.keys():
#            data[key] = np.exp(data[key]) - shift
#    else:
#        data = np.exp(data) - shift
#    return data


def _apply_scale(data, scale):
    """Applies scale for preprocessing."""
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = (data[key] - scale["shift"]) * (1.0 / scale["mult"])
    else:
        data = (data - scale["shift"]) * (1.0 / scale["mult"])
    return data


def _apply_scale_inverse(data, scale):
    """Applies inverse scale for postprocessing."""
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = data[key] * scale["mult"] + scale["shift"]
    else:
        data = data * scale["mult"] + scale["shift"]
    return data


def preprocess_features(features, params, logger, scale=None, array_name="features"):
    # exit if nothing to do
    if dictarray_is_none(features):
        return None
    data_params = _resolve_data_params(params, "all")
    features_type = data_params["features_type"].casefold()
    # DEV
    #   # apply transformation
    #   if features_type == 'RATE_DURATION'.casefold():
    #       for key in features.keys():
    #           features[key][...,1] = _log_transform(features[key][...,1], shift=1.0)
    # /DEV
    # calculate scaling for normalization
    if scale is None:
        shape = features["train"].ndim * [1]
        shape[1] = features["train"].shape[1]
        dtype = features["train"].dtype
        scale = {
            "shift": np.zeros(shape, dtype=dtype),
            "mult": np.ones(shape, dtype=dtype),
        }
        if params["data"].get("features_normalize", False):
            if features_type in [
                "TIME".casefold(),
                "TIME_NOISE".casefold(),
                "NOISE".casefold(),
            ]:
                assert 3 == features["train"].ndim
                scale = {
                    "shift": np.mean(features["train"], axis=(0, 2), keepdims=True),
                    "mult": np.std(features["train"], axis=(0, 2), keepdims=True),
                }
            elif features_type in ["ODE_STATS".casefold(), "RATE_DURATION".casefold()]:
                assert 3 == features["train"].ndim
                assert 1 == features["train"].shape[1]
                scale = {
                    "shift": np.nanmean(features["train"], axis=0, keepdims=True),
                    "mult": np.nanstd(features["train"], axis=0, keepdims=True),
                }
            # DEV
            #               # override scaling of "spike rate"
            #               if 'RATE_DURATION'.casefold() == features_type:
            #                   features_min = np.nanmin(features['train'], axis=0, keepdims=True)
            #                   features_max = np.nanmax(features['train'], axis=0, keepdims=True)
            #                   scale['shift'][...,0] = features_min[...,0]
            #                   scale['mult'][...,0]  = features_max[...,0] - features_min[...,0]
            # /DEV
            else:
                raise NotImplementedError(f"Unknown {features_type=}")
    logger.info(f"{array_name} scale = {scale}")
    if 3 < features["train"].ndim:
        logger.warning(
            f"Scaling of features is currently only tested for ndim=3,"
            + f" got ndim={features['train'].ndim}"
        )
    # apply scaling
    features = _apply_scale(features, scale)
    # replace nan values
    if features_type in [
        "TIME".casefold(),
        "TIME_NOISE".casefold(),
        "NOISE".casefold(),
    ]:
        pass
    elif features_type in ["ODE_STATS".casefold(), "RATE_DURATION".casefold()]:
        for key in features.keys():
            features[key] = np.where(np.isnan(features[key]), -10.0, features[key])
    else:
        raise NotImplementedError(f"Unknown {features_type=}")
    # return scale
    return scale


def postprocess_features(features, scale):
    # exit if nothing to do
    if dictarray_is_none(features):
        return
    # apply inverse scaling
    features = _apply_scale_inverse(features, scale)


# DEV
#   # apply inverse scaling
#   features_type = params["data"]["features_type"].casefold()
#   if features_type == 'RATE_DURATION'.casefold():
#       features = _apply_scale_inverse(features, scale)
#       #TODO apply inverse transforms
#       raise NotImplementedError()
# /DEV


def preprocess_targets(targets, params, logger, scale=None, array_name="targets"):
    # exit if nothing to do
    if dictarray_is_none(targets):
        return None
    # calculate scaling for normalization
    if scale is None:
        shape = (1, *targets["train"].shape[1:])
        dtype = targets["train"].dtype
        scale = {
            "shift": np.zeros(shape, dtype=dtype),
            "mult": np.ones(shape, dtype=dtype),
        }
        if params["data"].get("targets_normalize", False):
            assert 1 < targets["train"].ndim
            scale = {
                "shift": np.mean(targets["train"], axis=0, keepdims=True),
                "mult": np.std(targets["train"], axis=0, keepdims=True),
            }
    logger.info(f"{array_name} scale = {scale}")
    if 2 < targets["train"].ndim:
        logger.warning(
            f"Scaling of targets is currently only tested for ndim=2,"
            + f" got ndim={targets['train'].ndim}"
        )
    # apply scaling
    targets = _apply_scale(targets, scale)
    # output scale
    return scale


def postprocess_targets(targets, scale):
    # exit if nothing to do
    if dictarray_is_none(targets):
        return
    # apply inverse scaling
    targets = _apply_scale_inverse(targets, scale)


# ---------------------------------------


def _get_positions_from_histogram(data, range, n_bins, relevant_bins_threshold):
    hist, bin_edges = np.histogram(data.flatten(), range=range, bins=n_bins)
    relevant_bin_indices = hist > relevant_bins_threshold
    relevant_bin_edges = (bin_edges[:-1])[relevant_bin_indices]
    n_relevant_bins = np.sum(relevant_bin_indices)
    if 10 < n_relevant_bins:
        cond_positions = np.linspace(relevant_bin_edges[0], relevant_bin_edges[-1], 5)
    else:
        cond_positions = relevant_bin_edges
    return cond_positions


def get_conditional_positions(features: np.ndarray, params):
    """Tuned for data set 2020-12-09."""
    data_dir = params["data"]["data_dir"]
    data_params = _resolve_data_params(params, "all")
    features_type = data_params["features_type"].casefold()
    # set function parameters
    fn_params = {
        "n_bins": {
            "TIME": None,
            "RATE": 1000,
            "DURATION": 25,
        },
        "range": {
            "TIME": None,
            #           'RATE':     [ 0.55, 0.95],
            #           'DURATION': [-0.50, 0.50],
            "RATE": [-0.5, 1.0],
            "DURATION": [-0.5, 0.5],
        },
        "relevant_bins_threshold": {
            "TIME": None,
            "RATE": features.shape[0] * 0.06,
            "DURATION": features.shape[0] * 0.01,
        },
    }
    # extract conditional positions
    assert 1 < features.shape[0]
    cond_positions = list()
    if features_type in [
        "TIME".casefold(),
        "TIME_NOISE".casefold(),
        "NOISE".casefold(),
    ]:
        # set indices of samples as positions for conditionals
        cond_positions.append(
            np.arange(features.shape[0] - 6, features.shape[0], dtype=np.int32)
        )
    elif features_type == "ODE_STATS".casefold():
        raise NotImplementedError()
    elif features_type == "RATE_DURATION".casefold():
        if "2020" in data_dir:
            #           cond_positions.append(np.array([ 0.60,  0.70,  0.80,  0.90]))
            #           cond_positions.append(np.array([-0.36, -0.34, -0.26, -0.18, -0.17]))
            cond_positions.append(np.array([-0.4205, 0.1795, 0.7795]))
            cond_positions.append(np.array([-0.34, -0.3, -0.26, -0.22]))
        elif "2025" in data_dir:
            cond_positions.append(np.array([-0.419, 0.175, 0.769]))
            cond_positions.append(np.array([-0.5, -0.26, 0.22, 0.46]))
        else:  # otherwise find values from histogram
            for i, key in enumerate(["RATE", "DURATION"]):
                cond_positions.append(
                    _get_positions_from_histogram(
                        features[..., i],
                        fn_params["range"][key],
                        fn_params["n_bins"][key],
                        fn_params["relevant_bins_threshold"][key],
                    )
                )
    else:
        raise ValueError(f"Unknown features_type: {features_type}")
    return cond_positions


def _filter_samples(
    features: torch.Tensor,
    targets: torch.Tensor,
    position,
    threshold,
):
    # filter features
    if features.shape[-1] == len(position):
        # if as many positions as features: threshold positions across all features
        for i, (pos, thresh) in enumerate(zip(position, threshold)):
            features_ = features[..., i].flatten()
            idx_thresh = (pos - thresh < features_) & (features_ < pos + thresh)
            indices = idx_thresh if i == 0 else indices & idx_thresh
    elif 1 == len(position) and features.shape[1:] == position[0].shape[1:]:
        # if positions are samples: threshold by the normed distance to samples
        sample = position[0]
        thresh = threshold[0]
        assert 1 == sample.shape[0]
        distances = torch.linalg.norm(features - sample, dim=(1, 2))
        indices = torch.where(distances < thresh)[0]
    if 0 == torch.sum(indices):
        raise ValueError(f"empty filtering result")
    features_filtered = features[indices]
    # apply filter to targets
    if 2 == targets.ndim:
        targets_filtered = targets[indices]
    elif 3 == targets.ndim:
        targets_filtered = targets[:, indices, ...]
    else:
        raise NotImplementedError(f"targets.ndim={targets.ndim}")
    return features_filtered, targets_filtered


def get_conditional_samples(
    features: torch.Tensor,
    targets: torch.Tensor,
    position,
    params,
):
    data_params = _resolve_data_params(params, "all")
    features_type = data_params["features_type"].casefold()
    # extract conditional samples
    assert 1 < features.shape[0]
    if features_type in [
        "TIME".casefold(),
        "TIME_NOISE".casefold(),
        "NOISE".casefold(),
    ]:
        threshold = [0.01 * features[0].numel()]
        features_cond, targets_cond = _filter_samples(
            features, targets, position, threshold
        )
    elif features_type == "ODE_STATS".casefold():
        raise NotImplementedError()
    elif features_type == "RATE_DURATION".casefold():
        # threshold = [0.05, 0.15]
        threshold = [0.6, 0.6]
        features_cond, targets_cond = _filter_samples(
            features, targets, position, threshold
        )
    else:
        raise ValueError(f"Unknown features_type: {features_type}")
    return features_cond, targets_cond


# ---------------------------------------


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
        features_sub_begin_sequence=None,
        features_sub_step=None,
        noise_idx_random=True,
        item_return_order="yx",
    ):
        super().__init__()
        assert features is not None or features_noise is not None
        assert targets is None or features is not None
        assert targets_noise is None or features_noise is not None
        assert features_sub_begin_sequence is None or not features_sub_begin_random
        assert features_sub_begin_sequence is None or 0 < len(
            features_sub_begin_sequence
        )

        # set arrays from arguments
        if features is not None:
            self.features = (
                features
                if isinstance(features, torch.Tensor)
                else torch.from_numpy(features)
            )
        else:
            self.features = None
        if targets is not None:
            self.targets = (
                targets
                if isinstance(targets, torch.Tensor)
                else torch.from_numpy(targets)
            )
        else:
            self.targets = None
        if features_noise is not None:
            self.features_noise = (
                features_noise
                if isinstance(features_noise, torch.Tensor)
                else torch.from_numpy(features_noise)
            )
        else:
            self.features_noise = None
        if targets_noise is not None:
            self.targets_noise = (
                targets_noise
                if isinstance(targets_noise, torch.Tensor)
                else torch.from_numpy(targets_noise)
            )
        else:
            self.targets_noise = None

        # set the size of the dataset (not including multiple features_sub_begin_sequence)
        if self.features is not None:
            self.dataset_size = self.features.size(0)
            assert self.targets is None or self.targets.size(0) == self.dataset_size
        elif self.features_noise is not None:
            self.dataset_size = self.features_noise.size(0)
            assert (
                self.targets_noise is None
                or self.targets_noise.size(0) == self.dataset_size
            )
        else:
            raise NotImplementedError()

        # set from arguments
        self.features_additive_noise_std = features_additive_noise_std
        self.features_transform_fn = features_transform_fn
        self.features_sub_length = features_sub_length
        self.features_sub_begin_random = features_sub_begin_random
        self.features_sub_begin_sequence = features_sub_begin_sequence
        self.noise_idx_random = noise_idx_random
        self.item_return_order = item_return_order.casefold()
        self.features_sub_step = features_sub_step

    def __len__(self):
        if self.features_sub_begin_sequence is not None:
            return self.dataset_size * len(self.features_sub_begin_sequence)
        return self.dataset_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.features_sub_begin_sequence is not None:
            if isinstance(idx, int):
                sub_idx = idx // self.dataset_size
                idx = idx % self.dataset_size
            else:
                sub_idx = [i // self.dataset_size for i in idx]
                idx = [i % self.dataset_size for i in idx]
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
        if features_transformed is not None and self.features_additive_noise_std:
            features_transformed += self.features_additive_noise_std * torch.randn(
                features_transformed.size()
            )
        # truncate features array
        if (
            self.features_sub_length
            and features is not None
            and self.features_sub_length < features.size(-1)
        ):
            assert features_transformed is not None
            K = self.features_sub_length
            if self.features_sub_begin_random:
                k = np.random.randint(features.size(-1) - self.features_sub_length)
                features = features[..., k : k + K]
                features_transformed = features_transformed[..., k : k + K]
            elif self.features_sub_begin_sequence is not None:
                if isinstance(idx, int):
                    assert isinstance(sub_idx, int)
                    k = self.features_sub_begin_sequence[sub_idx]
                    features = features[..., k : k + K]
                    features_transformed = features_transformed[..., k : k + K]
                else:
                    assert isinstance(sub_idx, list)
                    starts = [self.features_sub_begin_sequence[si] for si in sub_idx]
                    features = torch.stack(
                        [features[i][..., k : k + K] for i, k in enumerate(starts)]
                    )
                    features_transformed = torch.stack(
                        [
                            features_transformed[i][..., k : k + K]
                            for i, k in enumerate(starts)
                        ]
                    )
            else:
                features = features[..., :K]
                features_transformed = features_transformed[..., :K]
        # truncate features with step length
        if (
            self.features_sub_step
            and 1 < self.features_sub_step
            and features is not None
        ):
            assert features_transformed is not None
            features = features[..., :: self.features_sub_step]
            features_transformed = features_transformed[..., :: self.features_sub_step]
        # transform features
        if features_transformed is not None and self.features_transform_fn is not None:
            features_transformed = self.features_transform_fn(
                features_transformed[None, ...]
            )[0]
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
        if "xx".casefold() == self.item_return_order:
            assert targets is not None
            return (targets, targets)
        elif "xy".casefold() == self.item_return_order:
            assert features_transformed is not None and targets is not None
            return (targets, features_transformed)
        elif "yx".casefold() == self.item_return_order:
            assert features_transformed is not None and targets is not None
            return (features_transformed, targets)
        elif "yy".casefold() == self.item_return_order:
            assert features_transformed is not None and features is not None
            return (features_transformed, features)
        else:
            raise ValueError(f"Unknown item return order: {self.item_return_order}")


def create_dataloader(
    params,
    logger,
    mode,
    features,
    targets,
    features_noise,
    targets_noise,
    features_transform_fn=None,
):
    """Creates a PyTorch dataset and dataloader from numpy arrays.
    Ref: https://pytorch.org/docs/stable/data.html
    """
    mode_scope = "train" if mode.any(Mode.TRAIN | Mode.PROFILE) else "evaluate"
    mode_data_params = _resolve_data_params(params, mode_scope)
    features_additive_noise_std = mode_data_params.get(
        "features_additive_noise_std", 0.0
    )
    features_sub_length = mode_data_params.get("features_sub_length", 0)
    features_sub_begin_random = mode_data_params.get("features_sub_begin_random", False)
    features_sub_begin_sequence = mode_data_params.get("features_sub_begin_sequence")
    features_sub_step = mode_data_params.get("features_sub_step")
    item_return_order = params["dataloader"]["item_return_order"]

    if mode.any(Mode.TRAIN | Mode.PROFILE):
        shuffle = True
        batch_size = mode_data_params["train_batch_size"]
    elif mode.any(Mode.VALIDATE | Mode.PREDICT | Mode.EVAL):
        shuffle = False
        batch_size = mode_data_params["eval_batch_size"]
        if 0 < features_sub_length and features_sub_begin_random:
            features_sub_begin_random = False
            assert features_sub_begin_sequence is None
            interval = features.shape[-1] - features_sub_length
            features_sub_begin_sequence = [
                i for i in range(0, interval, features_sub_length // 2)
            ] + [interval]
    else:
        raise NotImplementedError()

    # create the dataset
    logger.info("New FHN_Dataset")
    dataset = FHN_Dataset(
        features,
        targets,
        features_noise=features_noise,
        targets_noise=targets_noise,
        features_additive_noise_std=features_additive_noise_std,
        features_transform_fn=features_transform_fn,
        features_sub_length=features_sub_length,
        features_sub_begin_random=features_sub_begin_random,
        features_sub_begin_sequence=features_sub_begin_sequence,
        features_sub_step=features_sub_step,
        noise_idx_random=shuffle,
        item_return_order=item_return_order,
    )

    # set arguments for dataloader
    dataloader_params = params["dataloader"]
    dataloader_config = DataLoaderConfig(
        shuffle=shuffle,
        drop_last=False,
        batch_size=batch_size,
        param_num_workers=dataloader_params.get("num_workers"),
        param_prefetch_factor=dataloader_params.get("prefetch_factor"),
        param_pin_memory=torch.accelerator.is_available(),
    )

    # create the dataloader
    logger.info(f"New dataloader with {dataloader_config}")
    dataloader = DataLoader(
        dataset,
        batch_size=dataloader_config.batch_size,
        shuffle=dataloader_config.shuffle,
        drop_last=dataloader_config.drop_last,
        num_workers=dataloader_config.num_workers,
        pin_memory=dataloader_config.pin_memory,
        prefetch_factor=dataloader_config.prefetch_factor,
        persistent_workers=dataloader_config.persistent_workers,
        multiprocessing_context=dataloader_config.multiprocessing_context,
    )

    # output
    return dataloader
