"""
Enumerators
"""

import enum

class ModeKeys(enum.Enum):
    TRAIN    = enum.auto()
    VALIDATE = enum.auto()
    EVAL     = enum.auto()
    PREDICT  = enum.auto()

    @staticmethod
    def get_from_name(name):
        for mode in ModeKeys:
            if mode.name.casefold() == name.casefold():
                return mode
        raise ValueError('Unknown name for mode: '+name)


class NetworkType(enum.Enum):
    DENSENET       = enum.auto()
    CONVNET        = enum.auto()
    TRANSFORMERNET = enum.auto()

    @staticmethod
    def get_from_name(name):
        for mode in NetworkType:
            if mode.name.casefold() == name.casefold():
                return mode
        raise ValueError('Unknown name for model type: '+name)

###############################################################################

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

###############################################################################

"""
Plotting
"""

import matplotlib.pyplot as plt
import numpy as np

def save_loss(loss, path, plot_name, n_epochs,
              loss_std=None, x_offset=0, x_twin=False, y_scale='linear'):
    """ Saves raw loss values and plots these. """
    # save txt
    if loss_std is not None:
        np.savetxt(f"{path}.txt", np.stack((loss, loss_std), axis=-1))
    else:
        np.savetxt(f"{path}.txt", loss)
    # create plot
    fig, ax = plt.subplots(figsize=(6,4))
    if loss_std is not None:
        ax.errorbar(np.arange(x_offset, x_offset+len(loss)), loss, yerr=loss_std,
                    fmt='k-', ecolor='tab:gray', elinewidth=1.0)
    else:
        ax.plot(loss, 'k-')
    # set up x-axis
    ax.set_xlim([x_offset, len(loss)])
    if x_twin:
        ax_twin = ax.twiny()
        ax_twin.set_xlim([x_offset, n_epochs])
    else:
        xticks = ax.get_xticks()
        xticks_scaled = xticks * (n_epochs/(len(loss) + x_offset - 1))
        xticklabels = np.char.mod('%g', xticks_scaled)
        ax.set_xticks(xticks, labels=xticklabels)
        ax.set_xlim([-0.5, len(loss)])
    # set up x-axis
    ax.set_yscale(y_scale)
    # set labels
    if x_twin:
        ax.set_xlabel('logged step')
        ax_twin.set_xlabel('epoch')
    else:
        ax.set_xlabel('epoch')
    ax.set_title(plot_name)
    # set other
    ax.grid()
    fig.tight_layout()
    # save plot
    fig.savefig(f"{path}.pdf", dpi=300)

def save_data_vs_predict(data: list, predict: list, path,
                         plot_name=None, x_label=None, y_label=None):
    """ Saves raw true (data) values vs. predictions and plots these as scatter plots. """
    assert len(data) == len(predict)
    # save txt
    for i, (x_, y_) in enumerate(zip(data, predict)):
        np.savetxt(f"{path}_{i}.txt", np.stack((x_.squeeze(), y_.squeeze()), axis=-1))
    # set default labels
    if plot_name is None:
        plot_name = len(data)*['']
    if x_label is None:
        x_label = len(data)*['data value']
    if y_label is None:
        y_label = len(data)*['predicted value']
    # create plot
    n_plot_cols = len(data)
    fig, ax = plt.subplots(1, n_plot_cols, figsize=(4*n_plot_cols,3))
    for i, (x_, y_) in enumerate(zip(data, predict)):
        lim = [np.min(x_), np.max(x_)]
        ax[i].scatter(x_, y_, s=4**2, alpha=0.5)
        ax[i].plot(lim, lim, linewidth=3, linestyle='--', color='tab:orange')
        ax[i].set_xlabel(x_label[i])
        ax[i].set_ylabel(y_label[i])
        ax[i].set_title(plot_name[i])
        ax[i].grid()
    # set other
    fig.tight_layout()
    # save plot
    fig.savefig(f"{path}.pdf", dpi=300)

