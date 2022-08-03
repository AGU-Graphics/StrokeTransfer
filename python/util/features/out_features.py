import os

import numpy as np
from matplotlib import cm, pyplot as plt

from util.features.features import compute_features
from util.fig_util import font_setting, im_crop, plot_image, save_fig
from util.gbuffer import load_rgba, save_rgba
from util.logger import getLogger
from util.verbose import verbose_range

logger = getLogger(__name__)



def feature_key_maps():
    key_map = {"$I(p)$": "I", "$I_d(p)$": "I_d", "$I_s(p)$": "I_s",
               "$I_{\\nabla_2}(p)$": "nabla_I", "$K(p)$": "K", "$H(p)$": "H",
               "$D_S (p)$": "D_S",
               "$N_0$": "N_x", "$N_1$": "N_y", "$N_2$": "N_z"}
    return key_map



def internal_file(option, data_name, frame, format="png", dir_name=None):
    if dir_name is None:
        out_dir = f"{option.internal}/{data_name}"
    else:
        out_dir = f"{option.internal}/{dir_name}/{data_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{data_name}_{frame:03d}.{format}"
    return out_path


def out_mask_frame(A, option, frame, xlim=[0.0, 1.0], ylim=[0.0, 1.0]):
    """ Save mask file on the specified frame.

    Args:
        A: input mask image.
        option: RegressionOption/TransferOption.
        frame: target frame.
        xlim: x limits of the plot.
        ylim: y limits of the plot.
    """
    font_setting(font_family='Times New Roman', font_size=20)
    fig = plt.figure(figsize=(16, 16), linewidth=0)

    fig.patch.set_alpha(0)
    fig.tight_layout()
    ax = plt.subplot(1, 1, 1)

    A = np.dstack((A, A, A, A))
    logger.debug(f"A.shpae: {A.shape}")

    plot_image(A)
    im_crop(ax, A, xlim, ylim)

    mask_file = internal_file(option, data_name=f"mask", frame=frame, format="png", dir_name="features")
    save_fig(mask_file, transparent=False)
    plt.clf()
    plt.close()


def apply_mask_frame(option, img_file, frame):
    """ Apply mask for the output file.

    Args:
        option: RegressionOption/TransferOption.
        img_file: input/output file (.png) for mask image.
        frame: target frame.
    """
    fig_image = load_rgba(img_file)

    mask_file = internal_file(option, data_name=f"mask", frame=frame, format="png", dir_name="features")
    A_fig = load_rgba(mask_file)[:, :, 3]

    fig_image[:, :, 3] = A_fig

    save_rgba(img_file, fig_image)


def out_feature_frame(option, frame, include_color_bar=False,
                      xlim=[0.0, 1.0], ylim=[0.0, 1.0]):
    """ Save the visualization of proxy features.

    Args:
        option: RegressionOption/TransferOption.
        frame: target frame.
        include_color_bar: if true, include color bar on the plot.
        xlim: x limits of the plot.
        ylim: y limits of the plot.
    """
    scale = 1.0
    out_file = internal_file(option, data_name=f"K", frame=frame, format="png", dir_name="features")

    if os.path.exists(out_file) and not option.internal_overwrite:
        return

    features = compute_features(option, frame, scale)

    font_setting(font_family='Times New Roman', font_size=20)
    key_map = feature_key_maps()

    A = features.A
    out_mask_frame(A, option, frame, xlim=xlim, ylim=ylim)

    for i, key in enumerate(features.features.keys()):
        fig = plt.figure(figsize=(16, 16), linewidth=0)

        fig.patch.set_alpha(0)
        fig.tight_layout()
        ax = plt.subplot(1, 1, 1)

        Xi = features.features[key]

        plot_image(Xi, cmap=cm.magma, vmin=-1, vmax=1)
        if include_color_bar:
            plt.colorbar(ticks=[-1, 0, 1])

        im_crop(ax, features.features[key], xlim, ylim)

        feature_name = key_map[key]

        out_file = internal_file(option, data_name=f"{feature_name}", frame=frame, format="png", dir_name="features")
        save_fig(out_file, transparent=False)

        plt.clf()
        plt.close()

        apply_mask_frame(option, out_file, frame)


def output_features(option):
    """ Save the visualization of proxy features.

    Args:
        option: RegressionOption/TransferOption.
    """
    frames = option.frame_range()
    for frame in verbose_range(option.verbose, frames, desc=f"Output Features"):
        out_feature_frame(option, frame=frame)
