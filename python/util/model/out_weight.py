# MIT License
#
# Copyright (c) 2022  Hideki Todo, Kunihiko Kobayashi, Jin Katsuragi, Haruna Shimotahira, Shizuo Kaji, Yonghao Yue
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os

from matplotlib import cm, pyplot as plt

from util.canonical_sections.canonical_sections import \
    compute_canonical_sections
from util.canonical_sections.out_canonical_sections import canonical_key_maps
from util.features.features import compute_features
from util.features.out_features import apply_mask_frame
from util.fig_util import font_setting, plot_image, save_fig
from util.gbuffer import load_internal_orientation_frame
from util.model.out_models import load_model
from util.verbose import verbose_range


def internal_file(option, data_name, frame, format="png", dir_name=None):
    if dir_name is None:
        out_dir = f"{option.internal}/{data_name}"
    else:
        out_dir = f"{option.internal}/{dir_name}/{data_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{data_name}_{frame:03d}.{format}"
    return out_path


def out_weight_map_frame(option, model, frame):
    """ Save weight map for the specified frame.

    Args:
        option: RegressionOption/RegressionMultiOption.
        model: target vector field model.
        frame: target frame.
    """

    scale = 0.5
    features = compute_features(option, frame, scale)
    canonical_sections = compute_canonical_sections(option, frame, scale)

    for key in canonical_sections.canonical_sections.keys():
        model.set_orientation(key, canonical_sections.canonical_sections[key])

    for key in features.features.keys():
        model.set_feature(key, features.features[key])

    N = load_internal_orientation_frame(option, "Normal", frame=frame, scale=scale, format="png", dir_name="gbuffers")
    A = N[:, :, 3]

    model.set_alpha(A)
    weights = model.compute_weight_map()

    key_maps = canonical_key_maps()

    font_setting(font_family='Times New Roman', font_size=20)

    for key in weights.keys():
        W = weights[key]
        key_name = key_maps[key]

        fig = plt.figure(figsize=(16, 16))
        fig.patch.set_alpha(1)
        fig.tight_layout()
        ax = plt.subplot(1, 1, 1)
        plot_image(W, cmap=cm.PuOr, vmin=-1.0, vmax=1.0)

        out_file = internal_file(option, data_name=f"{key_name}", frame=frame, format="png", dir_name="weights")
        save_fig(out_file, transparent=True)

        plt.clf()
        plt.close()

        apply_mask_frame(option, out_file, frame)


def out_weight_map(option, model_file):
    """ Save the weight map for the given model.

    Args:
        option: RegressionOption/RegressionMultiOption/TransferOption.
        model_file: target model file.
    """
    model = load_model(model_file)
    frames = option.frame_range()
    for frame in verbose_range(option.verbose, frames, desc=f"Output Weight Field"):
        out_weight_map_frame(option, model, frame=frame)


def out_regression_weight_map(option):
    """ Save the weight map for the regression.

    Args:
        option: RegressionOption/RegressionMultiOption.
    """
    out_weight_map(option, option.model_output_orientation)


def out_transfer_weight_map(option):
    """ Save the weight map for the transfer.

    Args:
        option: TransferOption.
    """
    out_weight_map(option, option.model_orientation)
