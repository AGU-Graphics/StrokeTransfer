import os
import cv2
import numpy as np
import tqdm
from matplotlib import cm
from matplotlib import pyplot as plt

from util.canonical_sections.canonical_sections import \
    compute_canonical_sections
from util.fig_util import (draw_bg, get_color_list, im_crop, plot_image,
                           plot_vf_grid, save_fig)
from util.gbuffer import load_internal_orientation_frame, load_rgba, save_rgba
from util.normalize.norm import normalize_vector_image
from util.verbose import verbose_range


def canonical_key_maps():
    key_map = {'$I_{\\perp}$': 'I_d_perp', '$I_{\\parallel}$': 'I_d_parallel',
               '$N^V_{\\perp}$': "n_perp", '$N^V_{\\parallel}$': "n_parallel",
               '$o^{s}_{\\perp}$': "o_perp", '$o^{s}_{\\parallel}$': "o_parallel"}
    return key_map


def internal_file(option, data_name, frame, format="png", dir_name=None):
    if dir_name is None:
        out_dir = f"{option.internal}/{data_name}"
    else:
        out_dir = f"{option.internal}/{dir_name}/{data_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{data_name}_{frame:03d}.{format}"
    return out_path


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

    h, w = fig_image.shape[:2]

    if A_fig.shape[0] != h:
        A_fig = cv2.resize(A_fig, (w, h))

    A_fig = np.array(A_fig)
    A_fig = np.maximum(A_fig, fig_image[:, :, 0])

    A_fig[:10, :] = 0.0
    A_fig[-10:, :] = 0.0
    A_fig[:, :10] = 0.0
    A_fig[:, -10:] = 0.0

    fig_image[:, :, 3] = A_fig

    save_rgba(img_file, fig_image)


def out_canonical_sections_frame(option, frame, xlim=[0.0, 1.0], ylim=[0.0, 1.0]):
    """ Save the visualization of canonical sections.

    Args:
        option: RegressionOption/TransferOption.
        frame: target frame.
        xlim: x limits of the plot.
        ylim: y limits of the plot.
    """
    key_map = canonical_key_maps()

    out_file = internal_file(option, data_name=f"I_d_parallel", frame=frame, format="png",
                             dir_name="canonical_sections")
    if os.path.exists(out_file) and not option.internal_overwrite:
        return

    scale = 0.5

    vf_scale = 2.3
    width_scale = 0.65

    N = load_internal_orientation_frame(option, "Normal", frame=frame, scale=scale, format="png", dir_name="gbuffers")

    canonical_sections = compute_canonical_sections(option, frame=frame, scale=scale)

    for i, key in enumerate(canonical_sections.canonical_sections.keys()):
        key_name = key_map[key]
        fig = plt.figure(figsize=(16, 16))
        ax = plt.subplot(1, 1, 1)
        plot_image(N[:, :, 2], cmap=cm.bone, vmin=-0.1, vmax=0.9)

        vi = np.array(canonical_sections.canonical_sections[key])
        vi[:, :, 2] = 0.0
        vi[:, :, :3] = normalize_vector_image(vi[:, :, :3])

        vf_color = [1.0, 0.0, 0.0]

        plot_vf_grid(vi, s=15, color=vf_color, scale=vf_scale, width_scale=width_scale)
        draw_bg(N[:, :, 3], bg_color=[0.0, 0.0, 0.0])

        if xlim is not None:
            im_crop(ax, N, xlim, ylim)

        out_file = internal_file(option, data_name=f"{key_name}", frame=frame, format="png",
                                 dir_name="canonical_sections")
        save_fig(out_file, transparent=False)

        plt.clf()
        plt.close()

        apply_mask_frame(option, out_file, frame)


def output_canonical_sections(option):
    """ Save the visualization of canonical sections.

    Args:
        option: RegressionOption/TransferOption.
    """
    frames = option.frame_range()
    for frame in verbose_range(option.verbose, frames, desc=f"Output Canonical Sections"):
        out_canonical_sections_frame(option, frame=frame)
