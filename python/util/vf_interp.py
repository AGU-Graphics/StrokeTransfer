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


def evaluate_1form(V, F, X):
    EV, FE, EF = igl.edge_topology(V, F)
    P_12 = V[EV[:, 1]] - V[EV[:, 0]]
    X_12 = 0.5 * (X[EV[:, 1]] + X[EV[:, 0]])

    omega = np.einsum('ij,ij->i', P_12, X_12)
    return omega


def edge_mid(V, F, X):
    EV, FE, EF = igl.edge_topology(V, F)
    X_12 = 0.5 * (X[EV[:, 1]] + X[EV[:, 0]])
    return X_12


def interp_vf_i(omega_ijk, V_ijk, N, w_ijk, area):
    P_ijk = np.zeros((3, V_ijk.shape[1]))
    P_ijk[0, :] = np.cross(N, V_ijk[2, :] - V_ijk[1, :])
    P_ijk[1, :] = np.cross(N, V_ijk[0, :] - V_ijk[2, :])
    P_ijk[2, :] = np.cross(N, V_ijk[1, :] - V_ijk[0, :])

    C_ij = np.zeros((3, 1))
    C_ij[0] = omega_ijk[2] * w_ijk[2] - omega_ijk[0] * w_ijk[1]
    C_ij[1] = omega_ijk[0] * w_ijk[0] - omega_ijk[1] * w_ijk[2]
    C_ij[2] = omega_ijk[1] * w_ijk[1] - omega_ijk[2] * w_ijk[0]

    X_f = np.einsum('ij, ij->j', C_ij, P_ijk)
    X_f /= area
    return X_f


def interp_vf(C, V, F, W):
    N_f = igl.per_face_normals(V, F, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    areas = igl.doublearea(V, F)
    EV, FE, EF = igl.edge_topology(V, F)

    FE0 = EV[FE, 0]

    FE_flip = np.ones_like(FE)

    FE_flip[FE0 != F] = -1

    P_ijk = np.zeros((F.shape[0], 3, V.shape[1]))
    P_ijk[:, 0, :] = np.cross(N_f, V[F[:, 2]] - V[F[:, 1]])
    P_ijk[:, 1, :] = np.cross(N_f, V[F[:, 0]] - V[F[:, 2]])
    P_ijk[:, 2, :] = np.cross(N_f, V[F[:, 1]] - V[F[:, 0]])

    FE_C = C[FE] * FE_flip

    CW_ij = np.zeros((F.shape[0], 3))
    CW_ij[:, 0] = FE_C[:, 2] * W[:, 2] - FE_C[:, 0] * W[:, 1]
    CW_ij[:, 1] = FE_C[:, 0] * W[:, 0] - FE_C[:, 1] * W[:, 2]
    CW_ij[:, 2] = FE_C[:, 1] * W[:, 1] - FE_C[:, 2] * W[:, 0]

    X_f = np.einsum('ij, ijk->ik', CW_ij, P_ijk)
    X_f = np.einsum('i, ij->ij', 1.0 / areas, X_f)
    return X_f


def interp_points(V, F, W):
    FV = V[F]

    V_interp = np.einsum('ij, ijk->ik', W, FV)
    return V_interp


def triangle_weights(num_ws):
    w1s = np.linspace(0.0, 1.0, num_ws)[1:-1]
    w2s = np.linspace(0.0, 1.0, num_ws)[1:-1]
    w1, w2 = np.meshgrid(w1s, w2s)

    ids = w1 + w2 < 1.0
    w1 = w1[ids]
    w2 = w2[ids]
    w3 = 1.0 - w1 - w2

    return np.vstack((w1, w2, w3)).T
