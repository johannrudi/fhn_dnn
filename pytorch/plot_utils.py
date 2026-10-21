"""
Plotting
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def _scatter_groups(ax, x, y, groups, group_labels, s=2**2, alpha=0.5):
    """Scatter plot with one categorical color per group and a legend.

    Draws all points in one call, in a fixed random order, so that no group is
    hidden behind the groups drawn after it.
    """
    assert len(x) == len(y) == len(groups)
    order = np.random.default_rng(0).permutation(len(x))
    colors = [f"C{g % 10}" for g in groups[order]]
    sc = ax.scatter(x[order], y[order], c=colors, s=s, alpha=alpha)
    handles = [
        Line2D([], [], marker="o", linestyle="", color=f"C{g % 10}", label=label)
        for g, label in enumerate(group_labels)
    ]
    ax.legend(handles=handles, fontsize="small")
    return sc


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
    close_plot=True,
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
    # set up x-axis and its labels
    ax.set_xlim((x_offset, len(loss)))
    if x_twin:
        ax_twin = ax.twiny()
        ax_twin.set_xlim((x_offset, n_epochs))
        ax.set_xlabel("logged step")
        ax_twin.set_xlabel("epoch")
    else:
        xticks = ax.get_xticks()
        xticks_scaled = xticks * (n_epochs / (len(loss) + x_offset - 1))
        xticklabels = np.char.mod("%g", xticks_scaled)
        ax.set_xticks(xticks, labels=xticklabels)
        ax.set_xlim((-0.5, len(loss)))
        ax.set_xlabel("epoch")
    # set up y-axis
    ax.set_yscale(y_scale)
    # set title
    ax.set_title(plot_name)
    # set other
    ax.grid()
    fig.tight_layout()
    if save_plot:
        fig.savefig(f"{path}.pdf", dpi=300)
    if close_plot:
        plt.close(fig)


def plot_data_vs_qoi(
    data: list[np.ndarray],
    predict: list[np.ndarray],
    path,
    save_txt=True,
    save_plot=True,
    plot_name=None,
    x_label=None,
    y_label=None,
    line_x=None,
    line_y=None,
    groups=None,
    group_labels=None,
    close_plot=True,
):
    """Plots a list of quantities as scatter plots. Saves raw numbers to files.

    If `groups` (one integer group index per sample) and `group_labels` (one
    label per group) are given, the points are colored by group and the text
    files get the group index as a third column.
    """
    assert len(data) == len(predict)
    assert (groups is None) == (group_labels is None)
    if save_txt:
        for i, (x_, y_) in enumerate(zip(data, predict)):
            columns = [x_.squeeze(), y_.squeeze()]
            if groups is not None:
                columns.append(groups)
            np.savetxt(f"{path}_{i}.txt", np.stack(columns, axis=-1))
    # set defaults
    if plot_name is None:
        plot_name = len(data) * [""]
    if x_label is None:
        x_label = len(data) * ["data"]
    if y_label is None:
        y_label = len(data) * ["qoi"]
    # set the reference line apart from the group colors
    line_color = "tab:orange" if groups is None else "black"
    # create plot
    n_plots = len(data)
    fig, ax = plt.subplots(1, n_plots, figsize=(4 * n_plots, 3))
    if n_plots == 1:
        ax = [ax]
    for i, (x_, y_) in enumerate(zip(data, predict)):
        if groups is not None:
            _scatter_groups(ax[i], x_, y_, groups, group_labels)
        else:
            ax[i].scatter(x_, y_, s=2**2, alpha=0.5)
        if line_x is not None and line_y is not None:
            ax[i].plot(
                line_x[i], line_y[i], linewidth=3, linestyle="--", color=line_color
            )
        ax[i].set_xlabel(x_label[i])
        ax[i].set_ylabel(y_label[i])
        ax[i].set_title(plot_name[i])
        ax[i].grid()
    # set other
    fig.tight_layout()
    if save_plot:
        fig.savefig(f"{path}.pdf", dpi=300)
    if close_plot:
        plt.close(fig)


def plot_data_vs_predict(
    data: list[np.ndarray],
    predict: list[np.ndarray],
    path,
    save_txt=True,
    save_plot=True,
    plot_name=None,
    x_label=None,
    y_label=None,
    groups=None,
    group_labels=None,
    close_plot=True,
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
        groups=groups,
        group_labels=group_labels,
        close_plot=close_plot,
    )


def plot_data_vs_predict_error(
    data: list[np.ndarray],
    predict: list[np.ndarray],
    path,
    save_plot=True,
    plot_name=None,
    x_label=None,
    y_label=None,
    rel_error=False,
    scatter_color=None,
    groups=None,
    group_labels=None,
    close_plot=True,
):
    """Calculates and plots prediction errors.

    Points are colored either continuously by `scatter_color` (with a
    colorbar) or by group with `groups` and `group_labels` (with a legend),
    see `plot_data_vs_qoi`; the two cannot be combined.
    """
    assert len(data) == len(predict)
    assert (groups is None) == (group_labels is None)
    if scatter_color is not None and groups is not None:
        raise ValueError("scatter_color and groups cannot both be given")
    # set the reference line apart from the group colors
    line_color = "tab:orange" if groups is None else "black"
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
    if n_plots == 1:
        ax = [ax]
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
        elif groups is not None:
            sc = _scatter_groups(ax[i], x_, error, groups, group_labels)
        else:
            sc = ax[i].scatter(x_, error, s=2**2, alpha=0.5)
        ax[i].plot(lim, [0, 0], linewidth=3, linestyle="--", color=line_color)
        ax[i].set_xlabel(x_label[i])
        ax[i].set_ylabel(y_label[i])
        ax[i].set_title(plot_name[i])
        ax[i].grid()
    # set other
    fig.tight_layout()
    if save_plot:
        fig.savefig(f"{path}.pdf", dpi=300)
    if close_plot:
        plt.close(fig)


def plot_metrics_vs_checkpoint(
    epochs,
    mse_train,
    mae_train,
    r2_train,
    mse_validation,
    mae_validation,
    r2_validation,
    path,
    save_txt=True,
    save_plot=True,
    close_plot=True,
):
    """Plot overall MSE/MAE/R2 vs checkpoint epoch for train and validation.

    Draws both splits on shared axes (different color and marker) with a
    legend. MSE and MAE use log y-scale; R2 is limited to ``[0, 1]``. Writes
    one ``{path}.txt`` and one ``{path}.pdf``.

    Args:
        epochs: Checkpoint epoch numbers (x-axis), one value per checkpoint.
        mse_train: Overall train MSE scalar per checkpoint.
        mae_train: Overall train MAE scalar per checkpoint.
        r2_train: Overall train R2 scalar per checkpoint.
        mse_validation: Overall validation MSE scalar per checkpoint.
        mae_validation: Overall validation MAE scalar per checkpoint.
        r2_validation: Overall validation R2 scalar per checkpoint.
        path: Output path stem (no extension); writes ``.txt`` and/or ``.pdf``.
        save_txt: Write the raw column dump when True.
        save_plot: Write the PDF figure when True.
        close_plot: Call ``plt.close(fig)`` after saving when True.
    """
    n = len(epochs)
    assert n == len(mse_train) == len(mae_train) == len(r2_train)
    assert n == len(mse_validation) == len(mae_validation) == len(r2_validation)

    # save raw metrics (both splits)
    if save_txt:
        np.savetxt(
            f"{path}.txt",
            np.stack(
                (
                    epochs,
                    mse_train,
                    mae_train,
                    r2_train,
                    mse_validation,
                    mae_validation,
                    r2_validation,
                ),
                axis=-1,
            ),
            header="epoch mse_train mae_train r2_train mse_validation mae_validation r2_validation",
        )

    # series style: train vs validation
    series = (
        ("train", (mse_train, mae_train, r2_train), "o-", "C0"),
        ("validation", (mse_validation, mae_validation, r2_validation), "s--", "C1"),
    )
    panel_titles = ("MSE", "MAE", "R2")
    n_plots = len(panel_titles)
    fig, ax = plt.subplots(1, n_plots, figsize=(4 * n_plots, 3))
    for i, title in enumerate(panel_titles):
        for label, values, fmt, color in series:
            ax[i].plot(epochs, values[i], fmt, color=color, label=label)
        ax[i].set_xlabel("epoch")
        ax[i].set_title(title)
        ax[i].grid(True, which="both")
        ax[i].legend()
        if title in ("MSE", "MAE"):
            ax[i].set_yscale("log")
        elif title in ("R2"):
            ax[i].set_ylim(0.5, 1.0)
        else:
            raise NotImplementedError(f"got {title=}")
    fig.tight_layout()
    if save_plot:
        fig.savefig(f"{path}.pdf", dpi=300)
    if close_plot:
        plt.close(fig)
