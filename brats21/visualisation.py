import argparse
from typing import Tuple, Union
import torch
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from brats21 import utils as bu


def plot_patient_modalities(patient, idx=50, figsize=(12, 12), cmap="viridis"):
    """Plot all four modalities of a given patient path.

    Args:
        patient (Path): Path to patient
        idx (int, optional): Slice to plot. Defaults to 50.
        figsize (tuple, optional): Figure size. Defaults to (12, 12).
        cmap (string, optional): Colormap. Defaults to "viridis".

    Returns:
        (tuple): Figure and axs objects.
    """
    flair = bu.load_flair(patient)
    t1 = bu.load_t1(patient)
    t1ce = bu.load_t1ce(patient)
    t2 = bu.load_t2(patient)

    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(*figsize)

    flair_img = axs[0, 0].imshow(flair.dataobj[:, :, idx], cmap=cmap)
    fig.colorbar(flair_img, ax=axs[0, 0])
    axs[0, 0].set_title("Flair")

    t1_img = axs[0, 1].imshow(t1.dataobj[:, :, idx], cmap=cmap)
    fig.colorbar(t1_img, ax=axs[0, 1])
    axs[0, 1].set_title("T1")

    t1ce_img = axs[1, 0].imshow(t1ce.dataobj[:, :, idx], cmap=cmap)
    fig.colorbar(t1ce_img, ax=axs[1, 0])
    axs[1, 0].set_title("T1CE")

    t2_img = axs[1, 1].imshow(t2.dataobj[:, :, idx], cmap=cmap)
    fig.colorbar(t2_img, ax=axs[1, 1])
    axs[1, 1].set_title("T2")

    return fig, axs


def plot_nnunet_progress(
    data_dir: Union[str, Path],
    smooth_window: int = 51,
    show_pbar: bool = True,
    grid: bool = True,
    **kwargs
) -> Tuple:
    """Plot evaluation metrics for all plots in nnunet data directory.

    Args:
        data_dir: nnunet data directory path.
        smooth_window: Window size of smoothing done by a Savgol filter with 2nd degree polynomial.
        show_pbar: Wheather to show pregress bar for all subplots
        grid: Wheather to display grids.
        **kwargs: Passed to matplotlib.pyplot.plot function.
    """
    latest_models = bu.find_all_model_meta_paths(data_dir)

    if smooth_window % 2 == 0:
        smooth_window += 1
    if smooth_window < 2:
        smooth_window = 3

    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(12, 4)

    with tqdm(latest_models, desc="Model", disable=not show_pbar) as pbar:
        for model in pbar:
            plot_stuff = torch.load(model, map_location=torch.device("cpu"))[
                "plot_stuff"
            ]

            # val_losses_tr_mode is always empty listand not plotted.
            tr_losses, val_losses, val_losses_tr_mode, val_eval_metrics = plot_stuff

            axs[0].plot(
                savgol_filter(tr_losses, smooth_window, 2),
                label=model.parent.name,
                **kwargs
            )
            axs[0].set_title("Training Loss")
            axs[1].plot(
                savgol_filter(val_losses, smooth_window, 2),
                label=model.parent.name,
                **kwargs
            )
            axs[1].set_title("Validation Loss")
            axs[1].get_shared_y_axes().join(axs[0], axs[1])
            axs[1].set_yticklabels([])

            axs[2].yaxis.tick_right()
            axs[2].yaxis.set_ticks_position("right")
            axs[2].plot(
                savgol_filter(val_eval_metrics, smooth_window, 2),
                label=model.parent.name,
                **kwargs
            )
            axs[2].set_title("Validation Eval Metric")

            if grid:
                for i in range(3):
                    axs[i].grid()

    axs[2].legend(bbox_to_anchor=(1.2, 0.8))
    return fig, axs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", "--dd")
    parser.add_argument(
        "-output_path",
        "--op",
        help="Ouput path containing image name. E.g. ./images/plot will generate a plot.png.",
    )
    args = parser.parse_args()

    fig, axs = plot_nnunet_progress(args.dd)
    fig.savefig(args.op, bbox_inches="tight")
