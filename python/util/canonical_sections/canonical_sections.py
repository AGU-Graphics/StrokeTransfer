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

from util.features.grad_feature import grad_dir
from util.features.luminance_feature import lumiance_feature

from util.gbuffer import (load_frame_by_name, load_internal_frame_by_name,
                          load_internal_orientation_frame)
from util.model.vf_model import vf_scale_with_alpha
from util.normalize.norm import normalize_vector_image


def rotate90_canonical_sections(u_parallel):
    """ Rotate canonical section parallel => perp.

    Args:
        u_parallel: parallel direction of canonical section.

    Returns:
        u_perp: parallel direction of canonical section.
    """
    N = np.zeros((u_parallel.shape[0], u_parallel.shape[1], 3), dtype=np.float32)
    N[:, :, 2] = 1.0

    V_rot = np.cross(u_parallel[:, :, :3], N[:, :, :3])
    V_rot = np.dstack((V_rot, u_parallel[:, :, 3]))

    return V_rot


class CanonicalSectionSet:
    def __init__(self):
        self.canonical_sections = {}
        self.canonical_section_shape = None

    def set_orientation(self, key, u):
        self.canonical_sections[key] = u
        self.canonical_section_shape = u.shape

    def set_silhouette_orientation(self, key, o_s_parallel):
        """ Set silhouette_orientation with o_s_parallel.

        Args:
            key: canonical section id.
            o_s_parallel: canonical section for silhouette orientation.
        """
        A = o_s_parallel[:, :, 3]

        o_s_parallel[:, :, :3] = normalize_vector_image(o_s_parallel[:, :, :3])
        o_s_perp = rotate90_canonical_sections(o_s_parallel)

        o_s_parallel[:, :, :3] = - o_s_parallel[:, :, :3]
        o_s_parallel = vf_scale_with_alpha(o_s_parallel, A)
        o_s_perp = vf_scale_with_alpha(o_s_perp, A)

        self.canonical_sections[f"${key}" + r"_{\perp}$"] = o_s_perp
        self.canonical_sections[f"${key}" + r"_{\parallel}$"] = o_s_parallel

    def set_gradient_orientations(self, key, L, I):
        """ Set gradient orientations for the given scalar field.

        Args:
            key: canonical section id.
            L: input scalar field image to compute gradient orientations.
            I:

        Returns:

        """
        A = I[:, :, 3]

        u_parallel = grad_dir(L, I)

        u_parallel[:, :, :3] = normalize_vector_image(u_parallel[:, :, :3])
        u_parallel = vf_scale_with_alpha(u_parallel, A)
        u_perp = rotate90_canonical_sections(u_parallel)

        u_perp = vf_scale_with_alpha(u_perp, A)

        self.set_orientation(f"${key}" + r"_{\perp}$", u_perp)
        self.set_orientation(f"${key}" + r"_{\parallel}$", u_parallel)


def compute_canonical_sections(option, frame, scale):
    """ Compute canonical sections on the specified frame with the given settings.

    Args:
        option: RegressionOption/RegressionMultiOpetion/TransferOption.
        frame: target frame to compute features.
        scale: scale parameter for image size.

    Returns:
        canonical_sections: computed canonical sections with the given settings.
    """
    func = canonical_sections_func(with_iilumination=True,
                                   with_view=True,
                                   with_normal=True)
    return func(option, frame, scale)


def canonical_sections_func(with_iilumination=True,
                            with_view=True,
                            with_normal=True):
    def func(option, frame, scale):
        N = load_internal_orientation_frame(option, "Normal", frame=frame, scale=scale, format="png",
                                            dir_name="gbuffers")
        N[:, :, 1] *= -1.0

        diffuse = load_frame_by_name(option.diffuse_file_template, frame=frame, scale=scale)

        if diffuse is None:
            return None

        I_d = lumiance_feature(diffuse)

        canonical_sections = CanonicalSectionSet()

        if with_iilumination:
            canonical_sections.set_gradient_orientations("I", I_d, diffuse)

        if with_normal:
            canonical_sections.set_gradient_orientations("N^V", N[:, :, 2], diffuse)

        if with_view:
            Sil_dir = load_internal_orientation_frame(option, "o_s", frame=frame, scale=scale, dir_name="gbuffers")
            canonical_sections.set_silhouette_orientation("o^{s}", Sil_dir)

        return canonical_sections

    return func
