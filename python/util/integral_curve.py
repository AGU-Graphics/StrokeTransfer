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
import random

from util.normalize.normalize_position import normalize_point01
from util.stroke_polygon_generator import vertex_project
from util.verbose import verbose_range
from util.vf_interp import interp_vf_i


class IntegralCurveData:
    """

    Attributes:
        lines: (n, 3) vertices data.
        lines_normals: (n, 3) normal data.
        colors: (4) color info.
        widths:  (n, 1) width array.
        indexes: anchor point indices used for integral curves.
    """
    def __init__(self):
        self.lines = []
        self.lines_normals = []
        self.colors = []
        self.widths = []
        self.indexes = []


def math_intersection_of_two_lines(p, v, q, w):
    vw = np.array([v, -w]).T

    A = vw.T @ vw
    A += np.eye(A.shape[0])*1.0e-10
    b = vw.T @ (q - p)
    st = np.linalg.solve(A, b)
    s, t = st

    P = p + s * v
    Q = q + t * w
    return P, Q, round(s, 5), round(t, 5)


def search_connect_edge(search_point, search_vector, search_FE, search_F_idx, search_E_idx, mesh_data):
    search_FE_idx = -1
    next_p = np.zeros(3)
    param = np.zeros(2)
    for FE_idx in range(search_FE.shape[0]):
        if search_E_idx == search_FE[FE_idx]:
            continue
        if mesh_data.FE_flip[search_F_idx, FE_idx] == 1:
            E_point = mesh_data.V[mesh_data.EV[search_FE[FE_idx], 0]]
            E_vector = mesh_data.V[mesh_data.EV[search_FE[FE_idx], 1]] - mesh_data.V[mesh_data.EV[search_FE[FE_idx], 0]]
        else:
            E_point = mesh_data.V[mesh_data.EV[search_FE[FE_idx], 1]]
            E_vector = mesh_data.V[mesh_data.EV[search_FE[FE_idx], 0]] - mesh_data.V[mesh_data.EV[search_FE[FE_idx], 1]]
        _, E_p, v_param, e_param = math_intersection_of_two_lines(search_point, search_vector, E_point, E_vector)
        if not search_FE_idx == -1:
            if v_param >= 0.0 and 0.0 <= e_param <= 1.0 and param[0] <= v_param:
                next_p = E_p
                param = np.array([v_param, e_param])
                search_FE_idx = FE_idx
        else:
            if v_param >= 0.0 and 0.0 <= e_param <= 1.0:
                next_p = E_p
                param = np.array([v_param, e_param])
                search_FE_idx = FE_idx
    return next_p, param, search_FE_idx


def find_barycentric_coordinate_from_edge(search_FE, next_E_idx, param):
    w2 = -1
    w3 = -1
    if search_FE[0] == next_E_idx:
        w2 = 1 - param[1]
        w3 = 0
    elif search_FE[1] == next_E_idx:
        w2 = param[1]
        w3 = 1 - param[1]
    elif search_FE[2] == next_E_idx:
        w2 = 0
        w3 = param[1]
    w1 = 1.0 - (w2 + w3)
    barycentric_coordinate = np.array([w1, w2, w3], dtype=np.float32).reshape(-1, 3)
    return barycentric_coordinate


def search_anchor_points_within_image(image_data, anchor_mesh_data, F_anc_idx):
    within_image = True if 0 <= anchor_mesh_data.img_x[F_anc_idx] < image_data.img_width and 0 <= anchor_mesh_data.img_y[F_anc_idx] < image_data.img_height else False
    return within_image


def create_one_integral_curve(image_data, mesh_data, anchor_mesh_data, search_anchor_F_idx, camera_info):
    plots = []
    plots_normal = []

    search_F_idx = anchor_mesh_data.F_idx[search_anchor_F_idx]
    length = anchor_mesh_data.length[search_anchor_F_idx]
    length = 1.0 if length == 0 else length
    search_vector_angle_rad = np.deg2rad(anchor_mesh_data.vector_angle[search_anchor_F_idx])

    search_point = anchor_mesh_data.point[search_anchor_F_idx]
    search_E_idx = -1
    search_FE = mesh_data.FE[search_F_idx, :]
    search_barycentric_coordinate = anchor_mesh_data.barycentric_coordinate[search_anchor_F_idx].reshape(-1, 3)
    V_in_search_F = mesh_data.V[mesh_data.F[search_F_idx, :]].reshape(-1, 3)

    plots.append(search_point.tolist())
    plots_normal.append(mesh_data.N_f[search_F_idx, :].tolist())
    total_length = 0
    loop_num = 0
    while total_length < length:
        search_omega = mesh_data.omega[search_FE]
        search_omega_flip = search_omega * mesh_data.FE_flip[search_F_idx, :]

        search_vector = interp_vf_i(search_omega_flip.flatten(), V_in_search_F, mesh_data.N_f[search_F_idx, :], search_barycentric_coordinate.flatten(), mesh_data.areas[search_F_idx])
        b = np.cross(search_vector, mesh_data.N_f[search_F_idx, :])
        b_norm = b / np.linalg.norm(b)
        search_vector_norm = np.linalg.norm(search_vector)
        search_vector = (search_vector * np.cos(search_vector_angle_rad) + search_vector_norm * b_norm * np.sin(search_vector_angle_rad)) * length
        next_point, param, search_FE_idx = search_connect_edge(search_point, search_vector, search_FE, search_F_idx, search_E_idx, mesh_data)

        if search_FE_idx == -1:
            break
        else:
            next_E_idx = search_FE[search_FE_idx]
            previous_F_idx = search_F_idx
            for F_idx_in_search_EF in mesh_data.EF[next_E_idx, :]:
                if previous_F_idx != F_idx_in_search_EF:
                    search_F_idx = F_idx_in_search_EF

            search_point_proj = vertex_project(camera_info, search_point.reshape(1, -1))
            search_point_proj_norm = normalize_point01(search_point_proj.reshape(-1), image_data.img_width, image_data.img_height)
            next_point_proj = vertex_project(camera_info, next_point.reshape(1, -1))
            next_point_proj_norm = normalize_point01(next_point_proj.reshape(-1), image_data.img_width, image_data.img_height)
            length_between_search_and_next = np.sqrt((search_point_proj_norm[0] - next_point_proj_norm[0]) ** 2 + (search_point_proj_norm[1] - next_point_proj_norm[1]) ** 2)
            total_length += length_between_search_and_next

            if param[0] <= 1.0e-5:
                loop_num += 1
            else:
                loop_num = 0
            if loop_num == 10:
                break
            plots.append(next_point.tolist())
            plots_normal.append(mesh_data.N_e[next_E_idx, :].tolist())
            if search_F_idx == -1:
                break

        search_point = next_point
        search_E_idx = next_E_idx
        search_FE = mesh_data.FE[search_F_idx, :]
        search_barycentric_coordinate = find_barycentric_coordinate_from_edge(search_FE, search_E_idx, param)
        V_in_search_F = mesh_data.V[mesh_data.F[search_F_idx, :]].reshape(-1, 3)

    return plots, plots_normal


def append_one_integral_curve_data(integral_curve_datas, anchor_mesh_data, search_anchor_F_idx, plots, plots_normals):
    if len(plots) != 1:
        integral_curve_datas.indexes.append(search_anchor_F_idx)

        width = anchor_mesh_data.width[search_anchor_F_idx]
        width = sum(anchor_mesh_data.width) / anchor_mesh_data.width.shape[0] if width == 0 else width
        integral_curve_datas.widths.append(width)
        integral_curve_datas.colors.append(anchor_mesh_data.color[search_anchor_F_idx, :].tolist())

        integral_curve_datas.lines.append(plots)
        integral_curve_datas.lines_normals.append(plots_normals)

    return integral_curve_datas


def integral_curve_random(image_data, mesh_data, anchor_mesh_data, max_stroke_num, camera_info, texture, verbose):
    integral_curve_datas = IntegralCurveData()
    search_anchor_F_idx = -1
    anchor_F_idx_range = np.arange(np.array(anchor_mesh_data.F_idx).shape[0])
    for m in verbose_range(verbose, range(max_stroke_num)):
        texture_alpha = 1
        while texture_alpha != 0:
            if sum(anchor_mesh_data.flag) == 0:
                return integral_curve_datas

            search_anchor_F_idx = int(random.choices(anchor_F_idx_range, weights=anchor_mesh_data.flag)[0])
            anchor_mesh_data.flag[search_anchor_F_idx] = 0

            within_image = search_anchor_points_within_image(image_data, anchor_mesh_data, search_anchor_F_idx)
            if not within_image:
                continue
            texture_alpha = texture[anchor_mesh_data.img_y[search_anchor_F_idx], anchor_mesh_data.img_x[search_anchor_F_idx], 3]

        plots, plots_normals = create_one_integral_curve(image_data, mesh_data, anchor_mesh_data, search_anchor_F_idx, camera_info)
        integral_curve_datas = append_one_integral_curve_data(integral_curve_datas, anchor_mesh_data, search_anchor_F_idx, plots, plots_normals)

    return integral_curve_datas


def integral_curve_order(image_data, mesh_data, anchor_mesh_data, max_stroke_num, camera_info, previous_frame_index, verbose):
    integral_curve_datas = IntegralCurveData()
    for m in verbose_range(verbose, range(max_stroke_num)):
        search_anchor_F_idx = int(previous_frame_index[m])

        within_image = search_anchor_points_within_image(image_data, anchor_mesh_data, search_anchor_F_idx)
        if not within_image:
            continue

        plots, plots_normals = create_one_integral_curve(image_data, mesh_data, anchor_mesh_data, search_anchor_F_idx, camera_info)
        integral_curve_datas = append_one_integral_curve_data(integral_curve_datas, anchor_mesh_data, search_anchor_F_idx, plots, plots_normals)

    return integral_curve_datas
