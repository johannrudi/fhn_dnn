"""
Enumerators
"""

import enum

class ModeKeys(enum.Enum):
    TRAIN   = enum.auto()
    EVAL    = enum.auto()
    PREDICT = enum.auto()

    @staticmethod
    def get_from_name(name):
        for mode in ModeKeys:
            if mode.name.casefold() == name.casefold():
                return mode
        raise ValueError('Unknown name for mode: '+name)


class ModelType(enum.Enum):
    DENSENET = enum.auto()
    CONVNET  = enum.auto()

    @staticmethod
    def get_from_name(name):
        for mode in ModelType:
            if mode.name.casefold() == name.casefold():
                return mode
        raise ValueError('Unknown name for model type: '+name)

"""
Runtime parameters
"""

import os
import pathlib
import yaml

def load_parameters(filepath):
    """ Reads a yaml file and returns a dictionary.

    :param string filepath: Path to yaml file
    """
    # load yaml file
    with open(filepath, 'r') as f:
        params = yaml.safe_load(f)
    return params

def save_parameters(params, model_dir, filename="params.yaml"):
    """ Saves a dictionary to a file in the model_dir.

    :param dict params: dict we want to write to a file in model_dir
    :param string model_dir: Directory we want to write to
    :param string filename: Name of file in model_dir we want to save to.
    """
    if not model_dir:
        raise ValueError(
            "model_dir is not provided. For saving params, a user-defined"
            + " model_dir must be passed through the yaml file"
        )
    # set path and create subdirectories
    path = pathlib.Path(model_dir)/filename
    try:
        os.makedirs(str(path.parent), exist_ok=True)
    except OSError as error:
        raise ValueError(
            f"Invalid path {model_dir} provided. Check the model_dir path!"
        )
    # save yaml file
    with open(path, 'w+') as f:
        yaml.dump(params, f, default_flow_style=False)

def update_parameters_from_args(runconfig_params, args):
    """
    Sets command line arguments from arguments into parameters.

    :param dict params: runconfig dict we want to update
    :param argparse namespace args: Command line arguments
    """
    # copy arguments to parameters
    if args:
        for (k, v) in list(vars(args).items()):
            runconfig_params[k] = v if v is not None else runconfig_params.get(k)
    # Provisional handling of negative or 0 values. According to the estimator
    # source code passing negative or 0 steps raises an error in the estimator.
    # However, handling None in yaml is not straightforward. We have to pass
    # `null` there which is converted to None in python which is clumsy for users.
    if runconfig_params.get("eval_steps") is not None:
        if runconfig_params["eval_steps"] <= 0:
            runconfig_params["eval_steps"] = None

