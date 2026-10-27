"""
Handling of data.
"""

import json
import logging
import pathlib
from collections.abc import Callable
from typing import Any, Literal, TypeVar, assert_never, get_args

import dlk.opt.distributed as distributed
import numpy as np
import torch
from dlk.data.loader import DataLoaderConfig
from dlk.mode import Mode
from torch.utils.data import DataLoader, Dataset
from typing_extensions import TypeIs

# kind of data in a record's artifacts and scales
DataKind = Literal["params", "states", "stats"]

# name of a subset cut from the splits
SubsetName = Literal["train", "validation", "test"]

# ---------------------------------------
# DictArray
# ---------------------------------------

# a dictarray holds either arrays for all subsets, or None for all
DictArray = dict[SubsetName, np.ndarray] | dict[SubsetName, None]


def dictarray_empty() -> dict[SubsetName, None]:
    """Dictarray without arrays."""
    return {"train": None, "validation": None, "test": None}


def dictarray_set(
    arr_train: np.ndarray, arr_validation: np.ndarray, arr_test: np.ndarray
) -> dict[SubsetName, np.ndarray]:
    """Dictarray of the train, validation, and test subsets."""
    return {"train": arr_train, "validation": arr_validation, "test": arr_test}


def dictarray_is_none(
    arrays: DictArray | None,
) -> TypeIs[dict[SubsetName, None] | None]:
    """When it returns False, type checkers know that `arrays` holds arrays."""
    return (
        arrays is None
        or arrays["train"] is None
        or arrays["validation"] is None
        or arrays["test"] is None
    )


def dictarray_is_not_none(
    arrays: DictArray | None,
) -> TypeIs[dict[SubsetName, np.ndarray]]:
    """When it returns True, type checkers know that `arrays` holds arrays."""
    return not dictarray_is_none(arrays)


# ---------------------------------------
# Records
# ---------------------------------------


def _load_record(data_dir: str | pathlib.Path, pattern: str) -> dict[str, Any]:
    """Read the one record in `data_dir` whose file name matches `pattern`."""
    matches = sorted(pathlib.Path(data_dir).glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one {pattern} in {data_dir}, found {matches}"
        )
    with open(matches[0]) as file:
        return json.load(file)


def load_records(data_params: dict[str, Any]) -> dict[str, dict[str, Any] | None]:
    """Read the ODE record, and the noise record if noise features or targets are used."""
    features_type = data_params["features_type"].casefold()
    targets_type = data_params.get("targets_type", "N/A").casefold()
    features_have_noise = features_type in ("noise", "time_noise")
    targets_have_noise = targets_type in ("noise", "ode_noise")
    any_have_noise = features_have_noise or targets_have_noise

    data_dir = data_params["data_dir"]
    return {
        "ode": _load_record(
            data_dir, data_params.get("ode_record", "fhn_record*.json")
        ),
        "noise": (
            _load_record(data_dir, data_params.get("noise_record", "ar1_record*.json"))
            if any_have_noise
            else None
        ),
    }


# ---------------------------------------
# Loading
# ---------------------------------------


def _load_splits(
    data_dir: pathlib.Path, record: dict[str, Any], kind: DataKind
) -> dict[str, np.ndarray]:
    """Load the train and test splits of one data kind.

    Each split is one file in `data_dir`, named in
    `record["artifacts"][split][kind]`. Arrays are memory-mapped; rows are
    only read when sliced or computed on.
    """
    files = {
        split: record["artifacts"][split][kind]["file"] for split in ("train", "test")
    }
    return {
        split: np.load(data_dir / filename, mmap_mode="r")
        for split, filename in files.items()
    }


def _load_splits_fhn2020(
    data_dir: pathlib.Path, record: dict[str, Any], kind: DataKind
) -> dict[str, np.ndarray]:
    """Load the train and test splits of one data kind from the 2020 dataset.

    That dataset stores one file per column, listed in
    `record["artifacts"][kind]["files"]`, and a single file holds both
    splits: the first `record["config"]["splits"]["train"]` samples are the
    train split and the rest are the test split. A single listed file is used
    as is; several files are stacked along axis 1.
    """
    columns = [np.load(data_dir / file) for file in record["artifacts"][kind]["files"]]
    array = columns[0] if len(columns) == 1 else np.stack(columns, axis=1)
    n_train = record["config"]["splits"]["train"]
    return {"train": array[:n_train], "test": array[n_train:]}


def _get_subset_slices(
    n_train_split: int,
    Ntrain: int,
    Nvalidation: int,
    Ntest: int | None,
) -> tuple[slice, slice, slice]:
    """Rows of the train, validation, and test subsets within their splits.

    The train subset is the first `Ntrain` rows of the train split and the
    validation subset is its last `Nvalidation` rows, so the two never overlap.
    The test subset is the first `Ntest` rows of the test split, or all of
    them when `Ntest` is None.
    """
    return (
        slice(0, Ntrain),
        slice(n_train_split - Nvalidation, n_train_split),
        slice(0, Ntest),
    )


def _partition_splits(
    splits: dict[str, np.ndarray],
    Ntrain: int,
    Nvalidation: int,
    Ntest: int | None,
) -> dict[SubsetName, np.ndarray]:
    """Cut the train/validation/test subsets from the train and test splits."""
    assert "train" in splits and "test" in splits
    n_train_split = splits["train"].shape[0]
    n_test_split = splits["test"].shape[0]

    if Ntrain + Nvalidation > n_train_split:
        raise ValueError(
            f"Ntrain + Nvalidation = {Ntrain + Nvalidation} exceeds the"
            f" {n_train_split} samples of the train split"
        )
    if Ntest is not None and Ntest > n_test_split:
        raise ValueError(
            f"Ntest = {Ntest} exceeds the {n_test_split} samples of the test split"
        )

    train, validation, test = _get_subset_slices(
        n_train_split, Ntrain, Nvalidation, Ntest
    )
    return dictarray_set(
        splits["train"][train],
        splits["train"][validation],
        splits["test"][test],
    )


def _shape_features(
    arrays: dict[SubsetName, np.ndarray], kind: DataKind, data_params: dict[str, Any]
) -> dict[SubsetName, np.ndarray]:
    """Give features the (samples, channels, length) layout and apply the configured ranges.

    A 2D array gets a channel axis: (N, 2) stats become (N, 1, 2), and
    single-channel 2020 states become (N, 1, Nt). The channel and length
    ranges apply to states only.
    """
    channels_range = data_params.get("features_use_channels_range")
    length_range = data_params.get("features_use_length_range")
    for key, array in arrays.items():
        if array.ndim == 2:
            array = array[:, np.newaxis, :]
        if kind == "states":
            if channels_range is not None:
                start, stop = channels_range
                array = array[:, start:stop, :]
            if length_range is not None:
                start, stop = length_range
                array = array[:, :, start:stop]
        arrays[key] = array
    return arrays


def _shape_to_str(arrays: dict[Any, np.ndarray]) -> str:
    """Shapes of the arrays by key, for logging."""
    return ", ".join(f"{key} {array.shape}" for key, array in arrays.items())


# data kind in a record's artifacts, by features type
_FEATURES_TYPE_TO_KIND: dict[str, DataKind] = {
    "time": "states",
    "time_noise": "states",
    "noise": "states",
    "ode_stats": "stats",
    "rate_duration": "stats",
}
_TARGETS_TYPES = ("ode", "noise", "ode_noise", "n/a")


# load function by the `kind` of a record
_RECORD_KIND_TO_LOAD_SPLITS_FN = {
    "scientific-data-factory/record": _load_splits,
    "fhn_dnn/mock-record-2020": _load_splits_fhn2020,  # hand-written records
}


def load_arrays(
    data_params: dict[str, Any],
    records: dict[str, dict[str, Any] | None],
    logger: logging.Logger | None = None,
) -> tuple[DictArray, DictArray, DictArray, DictArray]:
    """Load the subsets of the features and targets that their types ask for."""
    # set up logger
    if logger is None:
        logger = logging.getLogger("load_arrays")

    # set options
    features_type = data_params["features_type"].casefold()
    targets_type = data_params.get("targets_type", "n/a").casefold()
    if features_type not in _FEATURES_TYPE_TO_KIND:
        raise ValueError(f"unknown {features_type=}")
    if targets_type not in _TARGETS_TYPES:
        raise ValueError(f"unknown {targets_type=}")
    data_dir = pathlib.Path(data_params["data_dir"])
    Ntrain = data_params["Ntrain"]
    Nvalidation = data_params.get("Nvalidation") or 0
    Ntest = data_params.get("Ntest")

    def load_and_partition(
        record: dict[str, Any] | None, kind: DataKind
    ) -> dict[SubsetName, np.ndarray]:
        """Load the splits of one data kind and cut the subsets from them."""
        assert record is not None
        # set the load function based on the record kind
        if record["kind"] not in _RECORD_KIND_TO_LOAD_SPLITS_FN:
            raise ValueError(f"unknown record kind {record['kind']!r}")
        load_splits = _RECORD_KIND_TO_LOAD_SPLITS_FN[record["kind"]]
        splits = load_splits(data_dir, record, kind)
        logger.debug(
            f"{record['factory']} {kind} via {load_splits.__name__}: {_shape_to_str(splits)}"
        )
        # create the dictarray for the dataset
        return _partition_splits(splits, Ntrain, Nvalidation, Ntest)

    # initialize dictarrays
    features: DictArray = dictarray_empty()
    targets: DictArray = dictarray_empty()
    features_noise: DictArray = dictarray_empty()
    targets_noise: DictArray = dictarray_empty()

    # load ODE features
    if features_type != "noise":
        kind = _FEATURES_TYPE_TO_KIND[features_type]
        features = _shape_features(
            load_and_partition(records["ode"], kind),
            kind,
            data_params,
        )
        logger.debug(f"features: {_shape_to_str(features)}")

    # load noise features
    if features_type in ("noise", "time_noise"):
        features_noise = _shape_features(
            load_and_partition(records["noise"], "states"),
            "states",
            data_params,
        )
        logger.debug(f"features_noise: {_shape_to_str(features_noise)}")

    # load ODE targets
    if targets_type in ("ode", "ode_noise"):
        targets = load_and_partition(records["ode"], "params")
        logger.debug(f"targets: {_shape_to_str(targets)}")

    # load noise targets
    if targets_type in ("noise", "ode_noise"):
        targets_noise = load_and_partition(records["noise"], "params")
        logger.debug(f"targets_noise: {_shape_to_str(targets_noise)}")

    return features, targets, features_noise, targets_noise


# keys that must be the same for training and evaluation, so they belong in the
# `data` block; `data_train` and `data_evaluate` hold only what differs
_DATA_ONLY_KEYS = (
    "features_type",
    "targets_type",
    "features_use_channels_range",
    "features_use_length_range",
    "features_sub_length",
    "features_sub_step",
    "Ntrain",
    "Nvalidation",
    "Ntest",
)


# old names of keys
_RENAMED_KEYS = {"Nvalidate": "Nvalidation"}


def _check_data_only_keys(params: dict[str, Any]) -> None:
    """Raise if a key of the `data` block is set in `data_train` or `data_evaluate`,
    or if a key still has its old name."""
    for block in ("data", "data_train", "data_evaluate"):
        for old, new in _RENAMED_KEYS.items():
            if old in params.get(block, {}):
                raise ValueError(f"rename `{old}` to `{new}` in `{block}`")
    for block in ("data_train", "data_evaluate"):
        misplaced = [key for key in _DATA_ONLY_KEYS if key in params.get(block, {})]
        if misplaced:
            raise ValueError(f"move {misplaced} from `{block}` to `data`")


def load_data(
    params: dict[str, Any], logger: logging.Logger
) -> tuple[
    DictArray, DictArray, DictArray, DictArray, dict[str, dict[str, Any] | None]
]:
    """Load the records and the subsets of the arrays they list.

    Also sets `num_features`, `num_targets`, and a default `Ntest` in
    `params["data"]`, when they are not set.

    Returns:
        `(features, targets, features_noise, targets_noise, records)`.
    """
    _check_data_only_keys(params)
    data_params = params["data"]

    # read records and load arrays
    records = load_records(data_params)
    features, targets, features_noise, targets_noise = load_arrays(
        data_params, records, logger
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
            params["data"].setdefault("Ntest", features["test"].shape[0])
        elif dictarray_is_not_none(features_noise):
            params["data"]["num_features"] = list(features_noise["train"].shape[1:])
            params["data"].setdefault("Ntest", features_noise["test"].shape[0])
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

    # print sample sizes (as `load_arrays` reads them; no `Ntest` means all)
    Ntest = data_params.get("Ntest")
    logger.info(f"Ntrain:    {data_params['Ntrain']}")
    logger.info(f"Nvalidation: {data_params.get('Nvalidation') or 0}")
    logger.info(f"Ntest:     {Ntest if Ntest is not None else 'all'}")

    # print data shapes
    logger.debug(f"num_features: {params['data']['num_features']}")
    logger.debug(f"num_targets:  {params['data']['num_targets']}")

    # return data
    return features, targets, features_noise, targets_noise, records


def load_timesteps(record: dict[str, Any], data_params: dict[str, Any]) -> np.ndarray:
    """Load the timesteps of the states that a record lists.

    The timesteps get the same length range as the features, so that they
    match the states that `load_arrays` returns.
    """
    data_dir = pathlib.Path(data_params["data_dir"])
    timesteps = np.load(data_dir / record["artifacts"]["timesteps"]["file"])
    length_range = data_params.get("features_use_length_range")
    if length_range is not None:
        start, stop = length_range
        timesteps = timesteps[start:stop]
    return timesteps


# ---------------------------------------
# Scaling
# ---------------------------------------

# Each record stores a scale per data kind, `scales[kind] = {"add", "mult"}`:
#   scaled   = (x + add) * mult
#   unscaled = scaled / mult - add
# Preprocessing applies it; postprocessing reads it again and undoes it.

# a numpy array or a tensor, for functions that take and return either
ArrayT = TypeVar("ArrayT", np.ndarray, torch.Tensor)


# shape of a scale, to broadcast against a subset of one data kind
_KIND_TO_SCALE_SHAPE: dict[DataKind, tuple[int, ...]] = {
    "params": (1, -1),
    "states": (1, -1, 1),
    "stats": (1, 1, -1),
}


def get_scale(
    record: dict[str, Any],
    kind: DataKind,
    channels_range: list[int] | None = None,
) -> dict[str, np.ndarray]:
    """Read the scale of one data kind from a record, shaped to broadcast against a subset.

    For states, pass the features' channel range so the scale matches them.
    """
    entry = record["scales"][kind]
    shape = _KIND_TO_SCALE_SHAPE[kind]
    assert "add" in entry and "mult" in entry

    # set the scale
    scale = {
        key: np.asarray(entry[key], dtype=np.float32).reshape(shape)
        for key in ("add", "mult")
    }

    # modify the scale if the chanels range is limited
    if kind == "states" and channels_range is not None:
        start, stop = channels_range
        scale = {key: value[:, start:stop, :] for key, value in scale.items()}

    return scale


def get_targets_scale(
    records: dict[str, dict[str, Any] | None],
    data_params: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Scale of the network outputs: the ODE params, then the noise params."""
    targets_type = data_params.get("targets_type", "n/a").casefold()
    if targets_type not in _TARGETS_TYPES:
        raise ValueError(f"unknown {targets_type=}")

    scales = []
    if targets_type in ("ode", "ode_noise"):
        assert records["ode"] is not None
        scales.append(get_scale(records["ode"], "params"))
    if targets_type in ("noise", "ode_noise"):
        assert records["noise"] is not None
        scales.append(get_scale(records["noise"], "params"))
    return {
        "add": np.concatenate([scale["add"] for scale in scales], axis=1),
        "mult": np.concatenate([scale["mult"] for scale in scales], axis=1),
    }


def scale_array(
    x: ArrayT, scale: dict[str, np.ndarray], *, is_std: bool = False
) -> ArrayT:
    """Scale on one numpy array or tensor.

    With `is_std`, only `mult` is applied, as for standard deviations.
    """
    if isinstance(x, torch.Tensor):
        mult = torch.as_tensor(scale["mult"], dtype=x.dtype, device=x.device)
        add = torch.as_tensor(scale["add"], dtype=x.dtype, device=x.device)
        return x * mult if is_std else (x + add) * mult
    mult, add = scale["mult"], scale["add"]
    return x * mult if is_std else (x + add) * mult


def apply_scale(
    arrays: dict[Any, ArrayT],
    scale: dict[str, np.ndarray],
    *,
    is_std: bool = False,
) -> None:
    """Applies scale for preprocessing, in place for each subset."""
    for key in arrays.keys():
        arrays[key] = scale_array(arrays[key], scale, is_std=is_std)


def scale_inverse_array(
    x: ArrayT, scale: dict[str, np.ndarray], *, is_std: bool = False
) -> ArrayT:
    """Undo `scale` on one numpy array or tensor.

    With `is_std`, only `mult` is undone, as for standard deviations.
    """
    if isinstance(x, torch.Tensor):
        mult = torch.as_tensor(scale["mult"], dtype=x.dtype, device=x.device)
        add = torch.as_tensor(scale["add"], dtype=x.dtype, device=x.device)
        return x / mult if is_std else x / mult - add
    mult, add = scale["mult"], scale["add"]
    return x / mult if is_std else x / mult - add


def apply_scale_inverse(
    arrays: dict[Any, ArrayT],
    scale: dict[str, np.ndarray],
    *,
    is_std: bool = False,
) -> None:
    """Applies inverse scale for postprocessing, in place for each entry."""
    for key in arrays.keys():
        arrays[key] = scale_inverse_array(arrays[key], scale, is_std=is_std)


def preprocess_features(
    features: DictArray,
    record: dict[str, Any],
    data_params: dict[str, Any],
) -> None:
    """Scale ODE features or noise features with the scale of a record.

    For noise features, pass the record that matches their role: the noise
    record when the noise is the features (features type NOISE), the ODE
    record when the noise is added to the ODE features (TIME_NOISE), so both
    are in the same units. The noise then also gets the features' `add`,
    which offsets their sum by `add * mult`; that offset is constant and
    accepted.

    Scales only when `features_normalize` is set; NaN values of stats are
    replaced either way.
    """
    # exit if nothing to do
    if dictarray_is_none(features):
        return
    kind = _FEATURES_TYPE_TO_KIND[data_params["features_type"].casefold()]

    # apply scaling
    if data_params.get("features_normalize", False):
        channels_range = data_params.get("features_use_channels_range")
        apply_scale(features, get_scale(record, kind, channels_range))

    # replace nan values
    if kind == "stats":
        total_min = np.nanmin(
            [np.nanmin(arr, axis=0, keepdims=True) for arr in features.values()],
            axis=0,
        )
        total_max = np.nanmax(
            [np.nanmax(arr, axis=0, keepdims=True) for arr in features.values()],
            axis=0,
        )
        replace_val = total_min - 0.1 * (total_max - total_min)
        for key in features.keys():
            features[key] = np.where(
                np.isnan(features[key]), replace_val, features[key]
            )


def postprocess_features(
    features: DictArray,
    record: dict[str, Any],
    data_params: dict[str, Any],
) -> None:
    """Undo the scaling of `preprocess_features`, if `features_normalize` is set."""
    # exit if nothing to do
    if dictarray_is_none(features) or not data_params.get("features_normalize", False):
        return
    kind = _FEATURES_TYPE_TO_KIND[data_params["features_type"].casefold()]

    # apply scaling
    channels_range = data_params.get("features_use_channels_range")
    apply_scale_inverse(features, get_scale(record, kind, channels_range))


def preprocess_targets(
    targets: DictArray,
    record: dict[str, Any],
    data_params: dict[str, Any],
) -> None:
    """Scale targets with the params scale of a record, if `targets_normalize` is set."""
    # exit if nothing to do
    if dictarray_is_none(targets) or not data_params.get("targets_normalize", False):
        return

    # apply scaling
    apply_scale(targets, get_scale(record, "params"))


def postprocess_targets(
    targets: DictArray,
    record: dict[str, Any],
    data_params: dict[str, Any],
) -> None:
    """Undo the scaling of `preprocess_targets`, if `targets_normalize` is set."""
    # exit if nothing to do
    if dictarray_is_none(targets) or not data_params.get("targets_normalize", False):
        return

    # apply scaling
    apply_scale_inverse(targets, get_scale(record, "params"))


# ---------------------------------------
# Representative Samples
# ---------------------------------------


def get_representative_samples(
    record: dict[str, Any],
    data_params: dict[str, Any],
    subset_name: SubsetName,
) -> np.ndarray:
    """Indices into one subset of the record's representative samples.

    The record lists representative samples by row of a split, for each of
    its own subsets. Those of the record's subset with the same name are
    taken; as the subsets cut by `Ntrain`, `Nvalidation`, and `Ntest` need not
    match the record's, only the rows inside the subset are kept, counted
    from the subset's first row.
    """
    n_train_split = record["config"]["splits"]["train"]
    n_test_split = record["config"]["splits"]["test"]
    train, validation, test = _get_subset_slices(
        n_train_split,
        data_params["Ntrain"],
        data_params.get("Nvalidation") or 0,
        data_params.get("Ntest"),
    )

    # rows of the subset within its split
    if subset_name == "train":
        rows = range(n_train_split)[train]
    elif subset_name == "validation":
        rows = range(n_train_split)[validation]
    elif subset_name == "test":
        rows = range(n_test_split)[test]
    else:
        assert_never(subset_name)

    # keep the representative rows that fall in the subset
    positions = []
    for row in record["index_sets"][subset_name]["representative"]["rows"]:
        if row in rows:
            positions.append(row - rows.start)
    return np.array(positions, dtype=np.int64)


# ---------------------------------------
# Sub-interval Begins
# ---------------------------------------


def eval_sub_begin_sequence(params: dict[str, Any]) -> list[int] | None:
    """The sub-interval begins of evaluation, `data_evaluate.features_sub_begin_sequence`."""
    return params["data_evaluate"].get("features_sub_begin_sequence")


def sub_begin_rows(
    n_rows: int, sub_begin_sequence: list[int] | None
) -> list[tuple[int | None, slice]]:
    """Rows of each sub-interval begin in the output of an unshuffled `FHN_Dataset`.

    With `features_sub_begin_sequence`, the dataset holds one contiguous block of
    `dataset_size` rows per begin, in the order of the sequence (the index
    mapping in `FHN_Dataset.__getitem__`). Without it, all rows belong to one
    begin, None.
    """
    if not sub_begin_sequence:
        return [(None, slice(0, n_rows))]
    n_begins = len(sub_begin_sequence)
    assert n_rows % n_begins == 0, f"{n_rows=} not a multiple of {n_begins=}"
    size = n_rows // n_begins
    return [
        (begin, slice(i * size, (i + 1) * size))
        for i, begin in enumerate(sub_begin_sequence)
    ]


def sub_begin_label(begin: int | None) -> str:
    """Label of a begin for logs and legends; empty for None."""
    return "" if begin is None else f"begin={begin}"


def sub_begin_suffix(begin: int | None) -> str:
    """Suffix of a begin for file names; empty for None, so names stay unchanged."""
    return "" if begin is None else f"_begin{begin}"


def sub_begin_groups(
    n_rows: int, sub_begin_sequence: list[int] | None
) -> tuple[np.ndarray | None, list[str] | None]:
    """Group index of each row and one label per begin, for coloring plots.

    Returns `(None, None)` without a sequence or rows.
    """
    if not sub_begin_sequence or n_rows == 0:
        return None, None
    groups = np.empty(n_rows, dtype=np.int64)
    labels = []
    for g, (begin, rows) in enumerate(sub_begin_rows(n_rows, sub_begin_sequence)):
        groups[rows] = g
        labels.append(sub_begin_label(begin))
    return groups, labels


# ---------------------------------------
# Dataset
# ---------------------------------------

# order of the pair a dataset item returns: x for targets, y for features
ItemReturnOrder = Literal["xx", "xy", "yx", "yy"]
_ITEM_RETURN_ORDERS: tuple[ItemReturnOrder, ...] = get_args(ItemReturnOrder)


class FHN_Dataset(Dataset):
    """Samples of features and targets, with optional noise, sub-intervals, and transform.

    With `features_sub_begin_sequence`, the dataset holds one contiguous block of
    rows per begin; `sub_begin_rows` gives the rows of each begin.
    """

    def __init__(
        self,
        features: np.ndarray | torch.Tensor | None,
        targets: np.ndarray | torch.Tensor | None,
        features_noise: np.ndarray | torch.Tensor | None = None,
        targets_noise: np.ndarray | torch.Tensor | None = None,
        features_additive_noise_std: float = 0.0,
        features_transform_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        features_sub_length: int | None = None,
        features_sub_begin_random: bool = False,
        features_sub_begin_sequence: list[int] | None = None,
        features_sub_step: int | None = None,
        noise_idx_random: bool = True,
        item_return_order: str = "yx",
    ) -> None:
        """Keep the arrays as tensors, and the options for drawing samples."""
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
        item_return_order = item_return_order.casefold()
        if item_return_order not in _ITEM_RETURN_ORDERS:
            raise ValueError(
                f"unknown {item_return_order=}, expected one of {_ITEM_RETURN_ORDERS}"
            )
        self.item_return_order: ItemReturnOrder = item_return_order
        self.features_sub_step = features_sub_step

    def __len__(self) -> int:
        """Number of samples, times the number of sub-interval starts if given."""
        if self.features_sub_begin_sequence is not None:
            return self.dataset_size * len(self.features_sub_begin_sequence)
        return self.dataset_size

    def __getitem__(
        self, idx: int | list[int] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One sample, or a batch for a list of indices, in `item_return_order`."""
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
        # NOTE: Do not add in place. `features_transformed` can be a view of
        #       the stored data, which may be a read-only memory map.
        if features_transformed is not None and self.features_additive_noise_std:
            features_transformed = (
                features_transformed
                + self.features_additive_noise_std
                * torch.randn(features_transformed.size())
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
                k = np.random.randint(features.size(-1) - self.features_sub_length + 1)
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
        if self.item_return_order == "xx":
            assert targets is not None
            return (targets, targets)
        elif self.item_return_order == "xy":
            assert features_transformed is not None and targets is not None
            return (targets, features_transformed)
        elif self.item_return_order == "yx":
            assert features_transformed is not None and targets is not None
            return (features_transformed, targets)
        elif self.item_return_order == "yy":
            assert features_transformed is not None and features is not None
            return (features_transformed, features)
        else:
            assert_never(self.item_return_order)


# ---------------------------------------
# Dataloader
# ---------------------------------------


def _resolve_batch_size(
    world_size: int,
    global_batch_size: int | None,
    local_batch_size: int | None,
) -> tuple[int, int]:
    """Turn whichever batch flag was given into a global and a local batch size."""

    if local_batch_size is not None:
        if global_batch_size is not None:
            raise ValueError("provide only one, global_batch_size or local_batch_size")
        global_batch_size = local_batch_size * world_size
    elif global_batch_size is not None:
        if global_batch_size % world_size != 0:
            raise ValueError(
                f"global batch size {global_batch_size} is not divisible by world size {world_size}"
            )
        local_batch_size = global_batch_size // world_size
    else:
        raise ValueError("expected either global_batch_size or local_batch_size")

    return global_batch_size, local_batch_size


def create_dataloader(
    params: dict[str, Any],
    logger: logging.Logger,
    mode: Mode,
    features: np.ndarray | torch.Tensor | None,
    targets: np.ndarray | torch.Tensor | None,
    features_noise: np.ndarray | torch.Tensor | None,
    targets_noise: np.ndarray | torch.Tensor | None,
    ctx: distributed.DistributedContext,
    *,
    features_transform_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    base_seed: int = 0,
) -> DataLoader:
    """Creates a PyTorch dataset and dataloader from numpy arrays.
    Ref: https://pytorch.org/docs/stable/data.html
    """
    # options shared by the modes, and options of this mode
    data_params = params["data"]
    if mode.any(Mode.TRAIN | Mode.PROFILE):
        mode_data_params = params["data_train"]
    else:
        mode_data_params = params["data_evaluate"]
    features_additive_noise_std = (
        mode_data_params.get("features_additive_noise_std") or 0.0
    )
    features_sub_length = data_params.get("features_sub_length") or 0
    features_sub_begin_random = (
        mode_data_params.get("features_sub_begin_random") or False
    )
    features_sub_begin_sequence = mode_data_params.get("features_sub_begin_sequence")
    features_sub_step = data_params.get("features_sub_step")
    item_return_order = params["dataloader"]["item_return_order"]

    if features_sub_length and not 1 < features_sub_length:
        raise ValueError(f"expected features_sub_length > 1, got {features_sub_length}")
    if features_sub_begin_random and features_sub_begin_sequence:
        raise ValueError(
            f"cannot set both features_sub_begin_random and features_sub_begin_sequence"
        )

    _, local_batch_size = _resolve_batch_size(
        ctx.world_size,
        mode_data_params.get("global_batch_size") or mode_data_params.get("batch_size"),
        mode_data_params.get("local_batch_size"),
    )

    if mode.any(Mode.TRAIN | Mode.PROFILE):
        noise_idx_random = mode_data_params.get("noise_idx_random") or True
        shuffle = True
        drop_last = True
    elif mode.any(Mode.VALIDATE | Mode.PREDICT | Mode.EVAL):
        noise_idx_random = mode_data_params.get("noise_idx_random") or False
        shuffle = False
        drop_last = False
        if features_sub_begin_random:
            logger.warning(
                f"features_sub_begin_random is active in a non-training mode"
            )
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
        noise_idx_random=noise_idx_random,
        item_return_order=item_return_order,
    )

    # create distributed data sampler (only for distributed training)
    if ctx.is_distributed:
        sampler = distributed.sampler_create(
            dataset, shuffle=shuffle, base_seed=base_seed, drop_last=drop_last
        )
    else:
        sampler = None

    # set arguments for dataloader
    dataloader_params = params["dataloader"]
    dataloader_config = DataLoaderConfig(
        shuffle=shuffle if sampler is None else False,
        drop_last=drop_last,
        batch_size=local_batch_size,
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
        sampler=sampler,
        drop_last=dataloader_config.drop_last,
        num_workers=dataloader_config.num_workers,
        pin_memory=dataloader_config.pin_memory,
        prefetch_factor=dataloader_config.prefetch_factor,
        persistent_workers=dataloader_config.persistent_workers,
        multiprocessing_context=dataloader_config.multiprocessing_context,
    )

    # output
    return dataloader
