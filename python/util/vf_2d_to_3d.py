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
from matplotlib import pyplot as plt

from util.rbf_model import *


def normalize_vectors(u):
    epsilon = 1e-10
    u_norm = np.sqrt(np.sum(u * u, axis=1))
    return np.einsum("ij,i->ij", u, 1.0 / (epsilon + u_norm))


def image_points(width, height):
    xs = range(width)
    ys = range(height)
    X, Y = np.meshgrid(xs, ys)
    P = np.dstack((X, Y))
    return P


def proj_vf_on_surface(u, N):
    u_dot_N = np.einsum("ij,ij->i", u, N)
    u_proj = u - np.einsum("i,ij->ij", u_dot_N, N)

    return u_proj


def project_3d_img(V, model_mat, view_mat, project_mat, I):
    h, w = I.shape[:2]
    viewport = np.array([0, h, w, -h])
    return project_3d_2d(V, model_mat, view_mat, project_mat, viewport)


def project_3d_2d(V, model_mat, view_mat, project_mat, viewport):
    MVP = model_mat @ view_mat @ project_mat
    V_3d = np.hstack((V, np.ones((V.shape[0], 1))))

    V_2d = V_3d @ MVP

    for i in range(3):
        V_2d[:, i] /= V_2d[:, 3]

    V_2d = 0.5 * V_2d + 0.5

    V_2d[:, 0] = viewport[0] + viewport[2] * V_2d[:, 0]
    V_2d[:, 1] = viewport[1] + viewport[3] * V_2d[:, 1]
    return V_2d


def unproject_scalar_by_image_sampling(I_img, V, model_mat, view_mat, project_mat):
    h, w = I_img.shape[:2]
    cs = int(I_img.size / (h * w))
    V_proj = project_3d_img(V, model_mat, view_mat, project_mat, I_img)
    V_proj = np.int32(V_proj)
    V_proj[:, 0] = np.clip(V_proj[:, 0], 0, w - 1)
    V_proj[:, 1] = np.clip(V_proj[:, 1], 0, h - 1)

    I_3d = np.zeros((V.shape[0], cs))

    for i, p in enumerate(V_proj):
        I_3d[i] = I_img[p[1], p[0]]
    return I_3d


def random_basis(V, N):
    B = np.random.rand(N.shape[0], 3)

    T = np.cross(B, N)
    B = np.cross(N, T)
    return T, B


def basis_weight(u_2D, t_2D, b_2D):
    tb = np.array([t_2D, b_2D]).T

    tb_inv = np.linalg.inv(tb)
    return tb_inv @ u_2D


def basis_weights(U_2D, T_2D, B_2D):
    AB = np.zeros((U_2D.shape[0], 2))
    for i in range(U_2D.shape[0]):
        AB[i, :] = basis_weight(U_2D[i, :], T_2D[i, :], B_2D[i, :])
    return AB


def unproj_vf_basis_weight(u_view, T, B, MVP):
    T_2D = T @ MVP
    B_2D = B @ MVP

    AB = basis_weights(u_view, T_2D, B_2D)

    u_3d = np.einsum("ij,i->ij", T, AB[:, 0].flatten()) + np.einsum("ij,i->ij", B, AB[:, 1].flatten())
    u_3d = normalize_vectors(u_3d)
    return u_3d


def unproject_vf_small(u_view, V, F, model_mat, view_mat, project_mat):
    N_V = igl.per_vertex_normals(V, F)
    MVP = MVP = model_mat @ view_mat @ project_mat
    MVP = MVP[:3, :2]

    T, B = random_basis(V, N_V)

    u_3d = unproj_vf_basis_weight(u_view, T, B, MVP)
    return u_3d


def unproject_vf_by_image_sampling(u_img, V, F, model_mat, view_mat, project_mat, proje_on_surf=False):
    N_V = igl.per_vertex_normals(V, F)
    u_view = unproject_scalar_by_image_sampling(u_img, V, model_mat, view_mat, project_mat)
    u_3d = unproject_vf_small(u_view[:, :2], V, F, model_mat, view_mat, project_mat)

    return u_3d
