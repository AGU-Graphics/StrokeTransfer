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

import cv2
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, Rbf, interp1d
from tqdm import tqdm

from util.fig_util import *
from util.gbuffer import (internal_file, load_internal_frame_by_name,
                          save_internal_frame_by_name)


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


def find_contours_from_bi(C_bi):
    contours, hierarchy = cv2.findContours(C_bi, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = [np.float32(P.reshape(-1, 2)) for P in contours]
    return contours


def find_contours(C):
    t, C_bi = cv2.threshold(np.uint8(255.0 * C), 127, 255, cv2.THRESH_BINARY)
    #     C_bi_not = cv2.bitwise_not(C_bi)
    #     skeleton =   cv2.ximgproc.thinning(C_bi_not, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    #     skeleton = cv2.bitwise_not(skeleton)

    return find_contours_from_bi(C_bi)


def find_silhouettes(A):
    t, A_bi = cv2.threshold(np.uint8(255.0 * A), 127, 255, cv2.THRESH_BINARY)
    return find_contours_from_bi(A_bi)


def compute_arc_length(P):
    E = P[1:, :] - P[:-1, :]
    len_strokes = np.sqrt(np.sum(E ** 2, axis=1))
    arc_parameter = np.zeros((P.shape[0]))
    arc_parameter[1:] = np.cumsum(len_strokes)
    return arc_parameter


def compute_arc_parameter(P):
    arc_parameter = compute_arc_length(P)
    arc_parameter /= arc_parameter[-1]
    return arc_parameter


def curve_func(P):
    t = compute_arc_parameter(P)
    fx = InterpolatedUnivariateSpline(t, P[:, 0])
    fy = InterpolatedUnivariateSpline(t, P[:, 1])

    def func(t_new):
        x = fx(t_new)
        y = fy(t_new)
        return np.dstack((x, y)).reshape(-1, 2)

    return func, t


def contour_directions(P0, dt=0.005, sp=30.0):
    if len(P0) < 2:
        return [], []
    try:
        f, t = curve_func(P0)
    except:
        return [], []
    L = compute_arc_length(P0)[-1]
    num_samples = max(int(L / sp), 6)
    t = np.linspace(0.0, 1.0, num_samples)
    P = f(t)
    t1 = np.clip(t + dt, 0, 1)
    t0 = np.clip(t - dt, 0, 1)
    u = f(t1) - f(t0)

    u = normalize_vectors(u)
    return P, u


def contour_drections_all(contours, sp=30.0):
    Qs = []
    us = []

    for Q0 in contours:
        Q0 = np.float32(Q0.reshape(-1, 2))
        Q, u = contour_directions(Q0, sp=sp)

        Qs.extend(Q)
        us.extend(u)
    Qs = np.array(Qs)
    us = np.array(us)
    return Qs, us


def plot_orientation(P, u, scale=50, color=None):
    # plt.plot(P[:, 0], P[:, 1], "o")
    plt.quiver(P[:, 0], P[:, 1], u[:, 0], -u[:, 1], scale=scale, color=color)


def draw_contours(contours):
    for P in contours:
        plt.plot(P[:, 0], P[:, 1], "-")


def compute_qualifying_score(vs, Ps, us, Qs, sigma_2):
    S = []
    for u, q in zip(us, Qs):
        Si = 0.0
        for v, p in zip(vs, Ps):
            d_qp = np.sum((q - p) ** 2)
            w = np.exp(-d_qp / (2 * sigma_2 ** 2))
            angle = np.dot(u, v)
            Si += w * angle

        S.append(Si)

    S = np.array(S)
    return S


def grid_points(width, height, num_grids=20):
    xs = np.linspace(0, width - 1, num_grids)
    ys = np.linspace(0, height - 1, num_grids)

    X, Y = np.meshgrid(xs, ys)
    P = np.dstack((X, Y))
    return P


def orientation_rbf(us0, Qs, S, smooth=1e-1):
    us = np.array(us0)
    id_neg = S <= 0.0
    us[id_neg, :] *= -1


    rbfs = []
    for i in range(2):
        rbf_i = Rbf(Qs[:, 0], Qs[:, 1], us[:, i], smooth=smooth)
        rbfs.append(rbf_i)

    def func(P):
        u_new = np.zeros((P.shape[0], 2))

        for i in range(2):
            u_new[:, i] = rbfs[i](P[:, 0], P[:, 1])
        u_new = normalize_vectors(u_new)
        return u_new

    return func


def find_direction_set(C):
    contours = find_contours(C[:, :, 0])
    silhouettes = find_silhouettes(C[:, :, 3])
    Ps, vs = contour_drections_all(silhouettes, sp=30.0)
    Qs, us = contour_drections_all(contours, sp=15.0)
    return Ps, vs, Qs, us


def find_silhouette_frame(scene_name, frame):
    scene_data = SceneDataset(scene_name)
    data_dir = scene_data.data_dir()

    gbuffer = GBuffer(data_dir)
    gbuffer.load_frame(frame, 1.0)
    C = gbuffer.suggestive_contour
    if C is None:
        return None, None
    silhouettes = find_silhouettes(C[:, :, 3])
    Ps, vs = contour_drections_all(silhouettes, sp=30.0)
    return Ps, vs


def rbf_interpolate(o_rbf, C):
    h, w = C.shape[:2]

    h_s, w_s = 512, 512

    P = np.float32(image_points(w_s, h_s))
    P = P.reshape(-1, 2)
    P[:, 0] *= w / float(w_s)
    P[:, 1] *= h / float(h_s)
    u_flat = o_rbf(P)
    u = u_flat.reshape(h_s, w_s, -1)
    u = 0.5 * u + 0.5
    u = np.clip(u, 0.0, 1.0)
    u = np.dstack((u, np.zeros_like(u[:, :, 0])))
    u = cv2.resize(u, (w, h))
    u_image = np.dstack((u, C[:, :, 3]))
    return u_image


def compute_orientation_frame_v2(scene_name, ch_name, frame, sigma_1, sigma_2, smooth=1e-1):
    scene_data = SceneDataset(scene_name)
    data_dir = scene_data.data_dir()

    gbuffer = GBuffer(data_dir)

    C = gbuffer.load_frame_by_name(ch_name, frame)

    Ps, vs, Qs, us = find_direction_set(C)

    S = compute_qualifying_score(vs, Ps, us, Qs, sigma_2)

    o_rbf = orientation_rbf(us, Qs, S, smooth)

    u_image = rbf_interpolate(o_rbf, C)
    gbuffer.save_frame_by_name(u_image, "SC_dir2", frame)


def compute_orientation_frame(scene_name, ch_name, frame, sigma_1, sigma_2, smooth=1e-1):
    scene_data = SceneDataset(scene_name)
    data_dir = scene_data.data_dir()

    gbuffer = GBuffer(data_dir)

    C = gbuffer.load_frame_by_name(ch_name, frame)

    Ps, vs, Qs, us = find_direction_set(C)

    Ps_frames = {}
    vs_frames = {}

    Ps_frames[frame] = Ps
    vs_frames[frame] = vs

    for i in range(int(frame - 2 * sigma_1), int(frame + 2 * sigma_1)):
        Ps_i, vs_i = find_silhouette_frame(scene_name, frame)
        if Ps_i is None:
            continue

        Ps_frames[i] = Ps_i
        vs_frames[i] = vs_i

    S = compute_qualifying_score_frames(vs_frames, Ps_frames, us, Qs, frame, sigma_1, sigma_2)

    o_rbf = orientation_rbf(us, Qs, S, smooth)

    u_image = rbf_interpolate(o_rbf, C)
    gbuffer.save_frame_by_name(u_image, "SC_dir", frame)


def save_RGBA(out_file, I):
    I_RGBA = np.uint8(255 * I)
    I_BGRA = cv2.cvtColor(I_RGBA, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(out_file, I_BGRA)


def compute_qualifying_score_frames(vs_frames, Ps_frames, us, Qs, frame, sigma_1, sigma_2):
    S = []

    for u, q in zip(us, Qs):
        Si = 0.0
        for i in Ps_frames.keys():
            Ps_i = Ps_frames[i]
            vs_i = vs_frames[i]
            for v, p in zip(vs_i, Ps_i):
                d_qp = np.sum((q - p) ** 2)
                d_f = (frame - i) ** 2
                w = np.exp(-d_f / (2 * sigma_1 ** 2)) * np.exp(-d_qp / (2 * sigma_2 ** 2))
            angle = np.dot(u, v)
            Si += w * angle

    S = np.array(S)
    return S


def u_at_P(u, P):
    us = []
    for p in P:
        p = np.int32(p)
        us.append(u[p[1], p[0], :])

    us = np.array(us)
    return us


def P_region(P0, A):
    alpha_ids = []
    for i, p in enumerate(P0):
        p = np.int32(p)
        if A[p[1], p[0]] > 0.5:
            alpha_ids.append(i)

    P = P0[alpha_ids, :]
    return P


def compute_silhouette_orientation_frame(option, frame=1, smooth=1e-1):
    data_name = "o_s"
    out_path = internal_file(option, data_name, frame=frame, dir_name="gbuffers")
    if os.path.exists(out_path) and not option.internal_overwrite:
        return

    C = load_internal_frame_by_name(option, "Normal", frame=frame, dir_name="gbuffers")
    u_image = np.zeros_like(C)

    silhouettes = find_silhouettes(C[:, :, 3])
    if len(silhouettes) > 0:
        Ps, vs = contour_drections_all(silhouettes, sp=5.0)

        if vs.shape[0] > 0:
            S = np.ones((Ps.shape[0], 1)).flatten()
            o_rbf = orientation_rbf(vs, Ps, S, smooth)
            u_image = rbf_interpolate(o_rbf, C)

    save_internal_frame_by_name(u_image, option, data_name, frame=frame, dir_name="gbuffers")
