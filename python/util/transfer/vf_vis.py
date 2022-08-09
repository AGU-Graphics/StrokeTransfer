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


import igl
import numpy as np

from pydec import simplex_quivers, simplicial_complex

from util.data_io.data_io import load_orientation, load_scene_data
from util.features.luminance_feature import luminance
from util.gbuffer import (load_frame_by_name, load_internal_frame_by_name,
                          load_internal_orientation_frame,
                          save_internal_frame_by_name)
from util.gl.renderer import Renderer
from util.normalize.norm import normalize_vectors
from util.verbose import verbose_range


def form_to_vf(sc, c_e):
    """ Decode vector field (Euclidean) from discrete 1-form.

    Args:
        sc: pydec.simplicial_complex object
        c_e: (#E, ) discrete 1-form for edge-based orientation field.

    Returns:
        V_F: (#F, 3) barycenter coordinates (Euclidean) of face-based vector field representation.
        u_F: (#F, 3) tangent vectors (Euclidean) assigned to V_F.
    """
    V_F, u_F = simplex_quivers(sc, c_e)
    return V_F, u_F


def out_vis_vf_frame(option, file_template, data_name="orientation", frame=1, color=[1.0, 0.0, 0.0, 1.0], grid_step=1):
    """ Render vector field result on the specified frame for visualization.

    Args:
        option: TransferOption/SmoothingOption includes vector field output.
        file_template: target vector field file template (.json) for visualization.
        data_name: output data name.
        frame: target frame for visualization.
        color: RGBA color used for vector field plot.
        grid_step: step size of image grid visualized in the results.
    """

    c_e = load_orientation(file_template, frame=frame)

    V, F, model_mat, view_mat, project_mat = load_scene_data(option, frame)
    N_F = igl.per_face_normals(V, F, np.array([0.0, 0.0, 1.0]))

    sc = simplicial_complex((V, F))

    V_F, u_F_dash = form_to_vf(sc, c_e)
    V_F += 0.01 * N_F
    u_F_dash = normalize_vectors(u_F_dash)

    vf_scale = 0.05
    scale = 1.0

    diffuse = load_frame_by_name(option.diffuse_file_template, frame=frame, scale=scale)
    I_d = luminance(diffuse)
    I_d /= np.max(I_d)
    N = load_internal_orientation_frame(option, "Normal", frame=frame, scale=scale, format="png", dir_name="gbuffers")

    rd = Renderer(im_width=N.shape[1], im_height=N.shape[0])
    rd.add_mesh(V, F, [0, 0, 0, 0])

    if grid_step > 1:
        fids = range(0, V_F.shape[0], grid_step)
        V_F = V_F[fids, :]
        u_F_dash = u_F_dash[fids, :]

    rd.lines = []
    rd.add_lines(V_F, V_F + vf_scale * u_F_dash, W=3.0, C=color)
    rd.setMVPMat(model_mat, view_mat, project_mat)

    I = rd.render()
    O = np.zeros_like(I)

    for ci in range(3):
        O[:, :, ci] = I[:, :, 3] * I[:, :, ci] + (1.0 - I[:, :, 3]) * I_d
    O[:, :, 3] = I[:, :, 3] + (1.0 - I[:, :, 3]) * diffuse[:, :, 3]
    save_internal_frame_by_name(O, option, data_name, frame=frame, scale=1, format="png", dir_name="view_orientations")
    return O


def out_vis_vf(option, file_template, data_name="orientation", color=[1.0, 0.0, 0.0, 1.0], grid_step=1):
    """ Render vector field results for visualization.

    Args:
        option: TransferOption/SmoothingOption includes vector field output.
        file_template: target vector field file template (.json) for visualization.
        data_name: output data name.
        color: RGBA color used for vector field plot.
        grid_step: step size of image grid visualized in the results.
    """
    frames = option.frame_range()

    for frame in verbose_range(option.verbose, frames, desc=f"Out Vector Field Image"):
        out_vis_vf_frame(option, file_template, data_name=data_name, frame=frame, color=color, grid_step=grid_step)
