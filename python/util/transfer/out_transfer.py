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


import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

from pydec import simplicial_complex

from util.base_option import FileTemplateOption
from util.canonical_sections.canonical_sections import \
    compute_canonical_sections
from util.data_io.data_io import load_scene_data, save_orientation
from util.features.features import compute_features
from util.features.out_features import apply_mask_frame
from util.fig_util import save_fig, plot_image
from util.gbuffer import (dilate_boundary, load_internal_orientation_frame,
                          save_frame_by_name, load_frame_by_name)
from util.model.out_models import load_model
from util.model.vf_model import proj_vf
from util.transfer.vf_vis import out_vis_vf
from util.verbose import verbose_range
from util.vf_2d_to_3d import unproject_vf_by_image_sampling

from util.logger import getLogger

logger = getLogger(__name__)


def vf_to_form(sc, u_V):
    """ Encode vector field (Euclidean) as 1-form.

    Args:
        sc: pydec.simplicial_complex object
        u_V: (#V, 3) tangent vectors (Euclidean) of vertex-based vector field representation.

    Returns:
        c_e: (#E, ) discrete 1-form for edge-based orientation field.
    """
    V = sc.vertices
    d0 = sc[0].d

    EV = sc[1].simplices

    P_ij = d0 * V
    u_ij_avg = 0.5 * (u_V[EV[:, 1]] + u_V[EV[:, 0]])

    c_e = np.einsum('ij,ij->i', P_ij, u_ij_avg)
    return c_e


def compute_1form_from_vf(option, frame, u0):
    """ Compute 1-form from the vector field image.

    Args:
        option: TransferOption.
        frame: target frame.
        u0: (h, w, d) vector field image.

    Returns:
        c_e: (#E, ) discrete 1-form for edge-based orientation field.
    """
    u = np.array(u0)
    u[:, :, 1] *= -1.0

    V, F, model_mat, view_mat, project_mat = load_scene_data(option, frame)

    u_V = unproject_vf_by_image_sampling(u[:, :, :3], V, F, model_mat, view_mat, project_mat,
                                         proje_on_surf=True)
    sc = simplicial_complex((V, F))
    c_e = vf_to_form(sc, u_V)
    return c_e


def out_1form(option, frame, u):
    """ Save 1-form data for orientation.

    Args:
        option:  TransferOption.
        frame: target frame.
        u: (h, w, d) vector field image.
    """
    c_e = compute_1form_from_vf(option, frame, u)

    save_orientation(option.output_orientation_template, c_e, frame=frame)


def out_vf_transfer_frame(option, model, frame):
    """ Save vector field transfer result on the specified frame.

    Args:
        option: TransferOption for transfer vector fields.
        model: VectorFieldRegressionModel for transfer.
        frame: target frame.
    """

    scale = 0.25
    N = load_internal_orientation_frame(option, "Normal", frame=frame, scale=scale, format="png", dir_name="gbuffers")

    N[:, :, 1] *= -1.0
    u = np.zeros_like(N)
    A = N[:, :, 3]
    if np.count_nonzero(A) > 0:
        canonical_sections = compute_canonical_sections(option, frame=frame, scale=scale)
        features = compute_features(option, frame=frame, scale=scale)

        for key in canonical_sections.canonical_sections.keys():
            model.set_orientation(key, canonical_sections.canonical_sections[key])

        for key in features.features.keys():
            model.set_feature(key, features.features[key])

        u = model.predict()
        u = proj_vf(u, N)

        for i in range(3):
            u[:, :, i] *= N[:, :, 3]

    out_1form(option, frame, u)


def out_vf_transfer(option):
    """ Save vector field transfer results.

    Args:
        option: TransferOption for transfer vector fields.
    """

    ## load model.
    model = load_model(option.model_orientation)

    ## vector field process loop.
    frames = option.frame_range()
    for frame in verbose_range(option.verbose, frames, desc=f"Vector Field Transfer"):
        out_vf_transfer_frame(option, model, frame)


def out_color_transfer(option):
    """ Save color transfer results.

    Args:
        option: TransferOption.
    """

    ## load model.
    color_model = load_model(option.model_color)

    ## color transfer process loop.
    frames = option.frame_range()
    scale = 1.0
    for frame in verbose_range(option.verbose, frames, desc=f"Color Transfer"):
        features = compute_features(option, frame=frame, scale=scale)

        for key in features.features.keys():
            color_model.set_feature(key, features.features[key])
        N = load_internal_orientation_frame(option, "Normal", frame=frame, scale=scale, format="png",
                                            dir_name="gbuffers")
        I_fit = color_model.predict(N.shape)

        A = I_fit[:, :, 3]
        I_fit[:, :, 3] = A * N[:, :, 3]

        save_frame_by_name(I_fit, option.output_color_template, frame=frame, format="png")


def vis_lw_transfer(option, transfer_name, out_template):
    frames = option.frame_range()

    file_template = FileTemplateOption(name="", root=None, file_template=out_template)
    png_template = out_template.replace(".exr", ".png")

    max_I = 0.0

    for frame in frames:
        N = load_internal_orientation_frame(option, "Normal", frame=frame, scale=1, format="png", dir_name="gbuffers")
        A = N[:, :, 3]
        I = load_frame_by_name(file_template, frame=frame, scale=1)

        max_I = max(max_I, np.max(I[A > 0.5]))

        logger.debug(f"vis_lw_transfer: max_I={max_I}")

    for frame in verbose_range(option.verbose, frames, desc=f"Visualize {transfer_name}"):
        I = load_frame_by_name(file_template, frame=frame, scale=1)

        fig = plt.figure(figsize=(16, 16), linewidth=0)

        fig.patch.set_alpha(0)
        fig.tight_layout()
        ax = plt.subplot(1, 1, 1)

        plot_image(I, cmap=cm.magma, vmin=0, vmax=max_I)

        out_file = png_template % frame
        save_fig(out_file, transparent=False)

        plt.clf()
        plt.close()

        apply_mask_frame(option, out_file, frame)
        # save_frame_by_name(I_fit, out_template, frame=frame, format="exr")


def out_lw_transfer(option, model_file, transfer_name, out_template):
    """ Save length/width transfer results.

    Args:
        option: TransferOption.
        model_file: model file (.pickle) for length/width transfer.
        transfer_name: process name for verbose.
        out_template: output image file template (.exr) for tranfer results.
    """

    ## load model.
    model = load_model(model_file)

    ## length/width transfer process loop.
    frames = option.frame_range()
    scale = 1.0
    for frame in verbose_range(option.verbose, frames, desc=transfer_name):
        features = compute_features(option, frame=frame, scale=scale)

        for key in features.features.keys():
            model.set_feature(key, features.features[key])

        I_fit = model.predict(features.feature_shape)
        I_fit = np.clip(I_fit, 0.0, 100.0)

        I_fit = dilate_boundary(I_fit, features.A)

        save_frame_by_name(I_fit, out_template, frame=frame, format="exr")


def out_transfer(option):
    """ Save transfer results of vector fields, color, length, width.

    Args:
        option: TransferOption for transfer vector fields, color, length, width.
    """

    ## vector field transfer.
    out_vf_transfer(option)
    out_vis_vf(option, option.output_orientation_template, data_name="orientation", color=[0.122, 0.467, 0.706, 1.0],
               grid_step=1)

    ## color transfer.
    out_color_transfer(option)

    ## length transfer.
    out_lw_transfer(option, option.model_length, "Length Transfer", option.output_length_template)

    ## width transfer.
    out_lw_transfer(option, option.model_width, "Width Transfer", option.output_width_template)


def vis_lw_transfers(option):
    vis_lw_transfer(option, "Length Transfer", option.output_length_template)
    vis_lw_transfer(option, "Width Transfer", option.output_width_template)
