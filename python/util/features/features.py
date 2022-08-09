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
from sklearn.preprocessing import MinMaxScaler

from util.features.curvature_feature import load_curvature
from util.features.ds_feature import ContourFieldBase
from util.features.grad_feature import load_grad
from util.features.luminance_feature import lumiance_feature
from util.fig_util import *
from util.gbuffer import (load_frame_by_name, load_internal_frame_by_name,
                          load_internal_orientation_frame)


class FeatureSet:
    def __init__(self):
        self.features = {}
        self.feature_shape = None

    def set_feature(self, key, feature):
        self.features[key] = feature
        self.feature_shape = feature.shape[:2]

    def set_alpha(self, A):
        self.A = A


def internal_file(option, data_name, frame, format="png"):
    out_dir = f"{option.internal}/{data_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{data_name}_{frame:03d}.{format}"
    return out_path


def IQR_q_abs(x, q=1):
    q1 = q
    x_abs = np.abs(x)
    q2 = 100 - q1
    q2, q1 = np.percentile(x_abs, [q2, q1])
    return q2


def frame_dependent_scaler(X, with_min_max_scaler=True):
    Y = {}
    for key in X.keys():
        Y[key] = np.array(X[key])

    K = X["$K(p)$"]
    H = X["$H(p)$"]

    R2_H = IQR_q_abs(H, q=1)
    R2_K = IQR_q_abs(K, q=1)
    R2 = max(R2_H, R2_K)
    K = np.array(np.clip(K, -R2_K, R2_K))
    H = np.array(np.clip(H, -R2_H, R2_H))

    scaler_curvature = MinMaxScaler(feature_range=(-1, 1))
    scaler_curvature.fit(np.array([-R2, R2]).reshape(-1, 1))

    K = np.array(np.clip(K, -R2_K, R2_K))
    H = np.array(np.clip(H, -R2_H, R2_H))

    K = scaler_curvature.transform(K.reshape(-1, 1)).reshape(K.shape)
    H = scaler_curvature.transform(H.reshape(-1, 1)).reshape(H.shape)

    nabla_I_d = X["$I_{\\nabla_2}(p)$"]

    g_std = np.std(nabla_I_d)
    g_mean = np.mean(nabla_I_d)
    nabla_I_d = np.clip(nabla_I_d, 0.0, g_mean + 2 * g_std)
    nabla_I_d /= g_mean + 2 * g_std

    Y["$K(p)$"] = K
    Y["$H(p)$"] = H
    Y["$I_{\\nabla_2}(p)$"] = nabla_I_d

    if with_min_max_scaler:
        scaler = MinMaxScaler(feature_range=(-1, 1))

        scaler_luminance = MinMaxScaler(feature_range=(-1, 1))
        scaler_luminance.fit(np.array([0, 1.39]).reshape(-1, 1))

        for key in X.keys():
            if key == "$K(p)$" or key == "$H(p)$":
                continue
            if key == "$I_d(p)$" or key == "$I_s(p)$":
                Y[key] = scaler_luminance.transform(Y[key].reshape(-1, 1)).reshape(Y[key].shape)

            else:
                Y[key] = scaler.fit_transform(Y[key].reshape(-1, 1)).reshape(Y[key].shape)
    return Y


def compute_features_func(with_iilumination=True, with_gradient=True,
                          with_curavtures=True, with_view=True, with_normal=True,
                          use_scaler=True):
    def func(option, frame, scale):

        N = load_internal_orientation_frame(option, "Normal", frame=frame, scale=scale, format="png", dir_name="gbuffers")
        A = N[:, :, 3]

        if np.count_nonzero(A) == 0:
            return None

        diffuse = load_frame_by_name(option.diffuse_file_template, frame=frame, scale=scale)
        specular = load_frame_by_name(option.specular_file_template, frame=frame, scale=scale)

        if diffuse is None:
            return None

        I_d = lumiance_feature(diffuse)
        I_s = lumiance_feature(specular)

        w = I_d.shape[1]

        K = load_curvature(option, ch_name="K", frame=frame, scale=scale)
        H = load_curvature(option, ch_name="H", frame=frame, scale=scale)

        nabla_I_d = load_grad(option, option.diffuse_file_template, frame=frame, scale=scale)

        cf_s = ContourFieldBase(N[:, :, 3], N, "Silhouette")

        D_S = cf_s.D_C0

        Xis = {}

        if with_iilumination:
            Xis["$I_d(p)$"] = I_d
            Xis["$I_s(p)$"] = I_s

        if with_gradient:
            Xis["$I_{\\nabla_2}(p)$"] = nabla_I_d

        if with_curavtures:
            Xis["$K(p)$"] = K
            Xis["$H(p)$"] = H

        if with_view:
            Xis["$D_S (p)$"] = D_S

        if with_normal:
            for ci in range(3):
                Xis[f"$N_{ci}$"] = N[:, :, ci]
                if ci == 2:
                    Xis[f"$N_{ci}$"] = np.clip(Xis[f"$N_{ci}$"], 0.0, 2.0)

        if use_scaler:
            Xis = frame_dependent_scaler(Xis, with_min_max_scaler=True)

        feature_set = FeatureSet()

        for key in Xis.keys():
            Xi = Xis[key]
            feature_set.set_feature(key, Xi)

        feature_set.set_alpha(A)

        return feature_set

    return func


def compute_features(option, frame, scale=1.0):
    """ Compute features on the specified frame with the given settings.

    Args:
        option: RegressionOption/RegressionMultiOpetion/TransferOption.
        frame: target frame to compute features.
        scale: scale parameter for image size.

    Returns:
        feature_set: computed features with the given settings.
    """
    func = compute_features_func(with_iilumination=True, with_gradient=True,
                                 with_curavtures=True, with_view=True, with_normal=True,
                                 use_scaler=True)
    return func(option, frame, scale)
