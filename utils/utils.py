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
    DENSERESNET    = enum.auto()
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

def save_parameters(params, save_dir, filename="params.yaml"):
    """ Saves a dictionary to a file in contained in save_dir.

    :param dict params: dict we want to write to a file in save_dir
    :param string save_dir: Directory we want to write to
    :param string filename: Name of file in save_dir we want to save to.
    """
    if not save_dir:
        raise ValueError(
            "save_dir is not provided. For saving params, a user-defined"
            + " save_dir must be passed through the yaml file"
        )
    # set path and create subdirectories
    path = pathlib.Path(save_dir)/filename
    try:
        os.makedirs(str(path.parent), exist_ok=True)
    except OSError as error:
        raise ValueError(
            f"Invalid path {save_dir} provided. Check the save_dir path!"
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

def plot_loss(loss, path, plot_name, n_epochs,
              loss_std=None, save_txt=True, save_plot=True, x_offset=0, x_twin=False, y_scale='linear'):
    """ Saves raw loss values and plots these. """
    if save_txt:
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
    if save_plot:
        fig.savefig(f"{path}.pdf", dpi=300)

def plot_data_vs_qoi(data: list, predict: list, path,
                     save_txt=True, save_plot=True, plot_name=None, x_label=None, y_label=None,
                     line_x=None, line_y=None):
    """ Plots a list of quantities as scatter plots. Saves raw numbers to files. """
    assert len(data) == len(predict)
    if save_txt:
        for i, (x_, y_) in enumerate(zip(data, predict)):
            np.savetxt(f"{path}_{i}.txt", np.stack((x_.squeeze(), y_.squeeze()), axis=-1))
    # set defaults
    if plot_name is None:
        plot_name = len(data)*['']
    if x_label is None:
        x_label = len(data)*['data']
    if y_label is None:
        y_label = len(data)*['qoi']
    # create plot
    n_plots = len(data)
    fig, ax = plt.subplots(1, n_plots, figsize=(4*n_plots,3))
    for i, (x_, y_) in enumerate(zip(data, predict)):
        ax[i].scatter(x_, y_, s=2**2, alpha=0.5)
        if line_x is not None and line_y is not None:
            ax[i].plot(line_x[i], line_y[i], linewidth=3, linestyle='--', color='tab:orange')
        ax[i].set_xlabel(x_label[i])
        ax[i].set_ylabel(y_label[i])
        ax[i].set_title(plot_name[i])
        ax[i].grid()
    # set other
    fig.tight_layout()
    if save_plot:
        fig.savefig(f"{path}.pdf", dpi=300)

def plot_data_vs_predict(data: list, predict: list, path,
                         save_txt=True, save_plot=True, plot_name=None, x_label=None, y_label=None):
    """ Saves raw true (data) values vs. predictions and plots these as scatter plots. """
    lim = list()
    for i, (x_, y_) in enumerate(zip(data, predict)):
        lim.append([np.min(x_), np.max(x_)])
    plot_data_vs_qoi(data, predict, path,
                     save_txt=save_txt, save_plot=save_plot,
                     plot_name=plot_name, x_label=x_label, y_label=y_label,
                     line_x=lim, line_y=lim)

def plot_data_vs_predict_error(data: list, predict: list, path,
                               save_plot=True, plot_name=None, x_label=None, y_label=None,
                               rel_error=False, scatter_color=None):
    """ Calculates and plots prediction errors. """
    assert len(data) == len(predict)
    # set defaults
    if plot_name is None:
        plot_name = len(data)*['']
    if x_label is None:
        x_label = len(data)*['data']
    if y_label is None:
        y_label = len(data)*['prediction error']
    # create plot
    n_plots = len(data)
    fig, ax = plt.subplots(1, n_plots, figsize=(4*n_plots,3))
    for i, (x_, y_) in enumerate(zip(data, predict)):
        lim = [np.min(x_), np.max(x_)]
        error = np.sqrt((x_ - y_)**2)
        if rel_error:
            error *= 1.0/np.sqrt(x_**2 + 1.0e-8)
        if scatter_color is not None and scatter_color[i] is not None:
            sc = ax[i].scatter(x_, error, c=scatter_color[i], cmap='viridis', s=2**2, alpha=0.5)
            fig.colorbar(sc, ax=ax[i])
        else:
            sc = ax[i].scatter(x_, error, s=2**2, alpha=0.5)
        ax[i].plot(lim, [0, 0], linewidth=3, linestyle='--', color='tab:orange')
        ax[i].set_xlabel(x_label[i])
        ax[i].set_ylabel(y_label[i])
        ax[i].set_title(plot_name[i])
        ax[i].grid()
    # set other
    fig.tight_layout()
    if save_plot:
        fig.savefig(f"{path}.pdf", dpi=300)
