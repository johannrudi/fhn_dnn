"""
Plotting
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_loss(
    loss,
    path,
    plot_name,
    n_epochs,
    loss_std=None,
    save_txt=True,
    save_plot=True,
    x_offset=0,
    x_twin=False,
    y_scale="linear",
):
    """Saves raw loss values and plots these."""
    if save_txt:
        if loss_std is not None:
            np.savetxt(f"{path}.txt", np.stack((loss, loss_std), axis=-1))
        else:
            np.savetxt(f"{path}.txt", loss)
    # create plot
    fig, ax = plt.subplots(figsize=(6, 4))
    if loss_std is not None:
        ax.errorbar(
            np.arange(x_offset, x_offset + len(loss)),
            loss,
            yerr=loss_std,
            fmt="k-",
            ecolor="tab:gray",
            elinewidth=1.0,
        )
    else:
        ax.plot(loss, "k-")
    # set up x-axis
    ax.set_xlim((x_offset, len(loss)))
    if x_twin:
        ax_twin = ax.twiny()
        ax_twin.set_xlim((x_offset, n_epochs))
    else:
        xticks = ax.get_xticks()
        xticks_scaled = xticks * (n_epochs / (len(loss) + x_offset - 1))
        xticklabels = np.char.mod("%g", xticks_scaled)
        ax.set_xticks(xticks, labels=xticklabels)
        ax.set_xlim((-0.5, len(loss)))
    # set up x-axis
    ax.set_yscale(y_scale)
    # set labels
    if x_twin:
        ax.set_xlabel("logged step")
        ax_twin.set_xlabel("epoch")
    else:
        ax.set_xlabel("epoch")
    ax.set_title(plot_name)
    # set other
    ax.grid()
    fig.tight_layout()
    if save_plot:
        fig.savefig(f"{path}.pdf", dpi=300)


def plot_data_vs_qoi(
    data: list,
    predict: list,
    path,
    save_txt=True,
    save_plot=True,
    plot_name=None,
    x_label=None,
    y_label=None,
    line_x=None,
    line_y=None,
):
    """Plots a list of quantities as scatter plots. Saves raw numbers to files."""
    assert len(data) == len(predict)
    if save_txt:
        for i, (x_, y_) in enumerate(zip(data, predict)):
            np.savetxt(
                f"{path}_{i}.txt", np.stack((x_.squeeze(), y_.squeeze()), axis=-1)
            )
    # set defaults
    if plot_name is None:
        plot_name = len(data) * [""]
    if x_label is None:
        x_label = len(data) * ["data"]
    if y_label is None:
        y_label = len(data) * ["qoi"]
    # create plot
    n_plots = len(data)
    fig, ax = plt.subplots(1, n_plots, figsize=(4 * n_plots, 3))
    for i, (x_, y_) in enumerate(zip(data, predict)):
        ax[i].scatter(x_, y_, s=2**2, alpha=0.5)
        if line_x is not None and line_y is not None:
            ax[i].plot(
                line_x[i], line_y[i], linewidth=3, linestyle="--", color="tab:orange"
            )
        ax[i].set_xlabel(x_label[i])
        ax[i].set_ylabel(y_label[i])
        ax[i].set_title(plot_name[i])
        ax[i].grid()
    # set other
    fig.tight_layout()
    if save_plot:
        fig.savefig(f"{path}.pdf", dpi=300)


def plot_data_vs_predict(
    data: list,
    predict: list,
    path,
    save_txt=True,
    save_plot=True,
    plot_name=None,
    x_label=None,
    y_label=None,
):
    """Saves raw true (data) values vs. predictions and plots these as scatter plots."""
    lim = list()
    for x_, _ in zip(data, predict):
        lim.append([np.min(x_), np.max(x_)])
    plot_data_vs_qoi(
        data,
        predict,
        path,
        save_txt=save_txt,
        save_plot=save_plot,
        plot_name=plot_name,
        x_label=x_label,
        y_label=y_label,
        line_x=lim,
        line_y=lim,
    )


def plot_data_vs_predict_error(
    data: list,
    predict: list,
    path,
    save_plot=True,
    plot_name=None,
    x_label=None,
    y_label=None,
    rel_error=False,
    scatter_color=None,
):
    """Calculates and plots prediction errors."""
    assert len(data) == len(predict)
    # set defaults
    if plot_name is None:
        plot_name = len(data) * [""]
    if x_label is None:
        x_label = len(data) * ["data"]
    if y_label is None:
        y_label = len(data) * ["prediction error"]
    # create plot
    n_plots = len(data)
    fig, ax = plt.subplots(1, n_plots, figsize=(4 * n_plots, 3))
    for i, (x_, y_) in enumerate(zip(data, predict)):
        lim = [np.min(x_), np.max(x_)]
        error = np.sqrt((x_ - y_) ** 2)
        if rel_error:
            error *= 1.0 / np.sqrt(x_**2 + 1.0e-8)
        if scatter_color is not None and scatter_color[i] is not None:
            sc = ax[i].scatter(
                x_, error, c=scatter_color[i], cmap="viridis", s=2**2, alpha=0.5
            )
            fig.colorbar(sc, ax=ax[i])
        else:
            sc = ax[i].scatter(x_, error, s=2**2, alpha=0.5)
        ax[i].plot(lim, [0, 0], linewidth=3, linestyle="--", color="tab:orange")
        ax[i].set_xlabel(x_label[i])
        ax[i].set_ylabel(y_label[i])
        ax[i].set_title(plot_name[i])
        ax[i].grid()
    # set other
    fig.tight_layout()
    if save_plot:
        fig.savefig(f"{path}.pdf", dpi=300)
