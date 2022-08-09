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


import cv2
import numpy as np
from cv2.ximgproc import guidedFilter
from matplotlib import cm
from matplotlib import pyplot as plt

from util.fig_util import plot_image, vf_show
from util.rbf_model import RBFModel


def normalize_vector(V):
    epsilon = 1e-10
    V_norm = epsilon + np.sqrt(np.einsum("ijk,ijk->ij", V, V))
    V_normalized = np.einsum("ijk, ij->ijk", V, 1.0 / V_norm)
    return V_normalized


def image_points(width, height):
    xs = range(width)
    ys = range(height)
    X, Y = np.meshgrid(xs, ys)
    P = np.dstack((X, Y))
    return np.float32(P)


def rot(V, N=None):
    if N is None:
        N = np.zeros((V.shape[0], V.shape[1], 3), dtype=np.float32)
        N[:, :, 2] = 1.0

    V_rot = np.cross(V[:, :, :3], N[:, :, :3])
    V_rot = normalize_vector(V_rot)
    V_rot = np.dstack((V_rot, V[:, :, 3]))

    return V_rot


def plot_vf_grid(V, s=20, color="red", scale=1.0):
    P = image_points(V.shape[1], V.shape[0])
    plt.quiver(P[::s, ::s, 0], P[::s, ::s, 1], scale * V[::s, ::s, 0], scale * V[::s, ::s, 1], color=color, angles='xy',
               scale=100.0)


def contour_distance_field(C):
    t, C_bin = cv2.threshold(np.uint8(255.0 * C[:, :, 0]), 127, 255, cv2.THRESH_BINARY)
    D_C = cv2.distanceTransform(C_bin, cv2.DIST_L2, 0)
    return D_C


def contour_orientation(C):
    D_C = contour_distance_field(C)
    u_x = cv2.Sobel(D_C, cv2.CV_64F, 1, 0, ksize=3)
    u_y = cv2.Sobel(D_C, cv2.CV_64F, 0, 1, ksize=3)

    u = np.zeros_like(C)
    u[:, :, 0] = u_x
    u[:, :, 1] = u_y

    u = normalize_vector(u)

    u[:, :, 3] = C[:, :, 3]
    return u


def contour_orientation_rbf(C):
    A = C[:, :, 3]

    I_C = np.ones_like(A)
    I_C = (1.0 - A) * I_C + A * C[:, :, 0]

    u_x = cv2.Sobel(I_C, cv2.CV_64F, 1, 0, ksize=3)
    u_y = cv2.Sobel(I_C, cv2.CV_64F, 0, 1, ksize=3)

    u_xy = np.dstack((u_x, u_y)).reshape(-1, 2)

    u_norm = np.linalg.norm(u_xy, axis=1)

    P = image_points(C.shape[1], C.shape[0]).reshape(-1, 2)

    rbf_model = RBFModel(k=10, smooth=1e-4)
    rbf_model.fit(P[u_norm > 0.001, :], u_xy[u_norm > 0.001, :])
    u_flat = rbf_model.transform(P)

    u = np.zeros_like(C)
    u[:, :, :2] = u_flat.reshape(u[:, :, :2].shape)

    u = normalize_vector(u)

    u[:, :, 3] = C[:, :, 3]
    return u


class ContourFieldBase:
    def __init__(self, C, N, label="Silhouette"):
        self.C = C
        self.N = N
        self.A = N[:, :, 3]
        self.label = label
        self.compute_distance_field()

    def compute_distance_field(self):
        C = self.C
        N = self.N

        t, C_bin = cv2.threshold(np.uint8(255.0 * C), 127, 255, cv2.THRESH_BINARY)
        D_C0 = cv2.distanceTransform(C_bin, cv2.DIST_L2, 0)
        D_C0_max = np.max(D_C0)

        if D_C0_max > 1e-10:
            D_C0 /= D_C0_max

        sigma_I = 1e-6
        sigma_s = 3

        DC = guidedFilter(N[:, :, :3], np.float32(D_C0), sigma_s, sigma_I)
        DC = np.min(D_C0) + (DC - np.min(DC)) / (np.max(D_C0) - np.min(D_C0))

        self.D_C0 = DC

    def compute_train_data(self, D_C0, N, num_samples=500):
        N_flat = N.reshape(-1, 4)
        A = N[:, :, 3]
        A_flat = N_flat[:, 3]
        P_flat = image_points(N.shape[1], N.shape[0]).reshape(-1, 2)
        P_flat /= np.max(P_flat)
        X = np.hstack((N_flat, P_flat))
        D_C0_flat = D_C0.flatten()

        X_sample = X[A_flat > 0.5, :]
        D_C_sample = D_C0_flat[A_flat > 0.5]

        if num_samples > 0:
            sample_ids = np.random.randint(X_sample.shape[0], size=num_samples)
            X_sample = X_sample[sample_ids, :]
            D_C_sample = D_C_sample[sample_ids]
        return X_sample, D_C_sample, X

    def compute_orientation(self, D_C):
        A = self.A
        u_x = cv2.Sobel(D_C, cv2.CV_64F, 1, 0, ksize=3)
        u_y = cv2.Sobel(D_C, cv2.CV_64F, 0, 1, ksize=3)

        h, w = A.shape[:2]

        u_grad = np.zeros((h, w, 4))
        u_grad[:, :, 0] = u_x
        u_grad[:, :, 1] = u_y

        u_grad = normalize_vector(u_grad)
        for ci in range(3):
            u_grad[:, :, ci] *= A
        u_grad[:, :, 3] = A

        u_rot = rot(u_grad, N=None)
        return {"D_C": D_C, "u_grad": u_grad, "u_rot": u_rot}

    def regression(self, D_C0, N, model, num_samples=500):
        N_flat = N.reshape(-1, 4)
        A = N[:, :, 3]
        X_sample, D_C_sample, X = self.compute_train_data(D_C0, N, num_samples)

        model.fit(X_sample, D_C_sample)
        D_C_flat = model.predict(X)
        D_C_flat = np.clip(D_C_flat, 0.0, 1.0)
        D_C = D_C_flat.reshape(D_C0.shape)
        D_C = D_C * A
        return D_C

    def plot(self):
        C = self.C

        fig = plt.figure(figsize=(16, 12))
        plt.subplot(2, 3, 1)
        plot_image(C)
        plt.title(self.label)

        self.plot_field(self.u_grad_rot, self.label)

    def plot_field(self, u_grad_rot, label):
        C = self.C

        plt.subplot(2, 3, 4)
        plot_image(u_grad_rot["D_C"], cmap=cm.magma)
        plt.title(f"Distance: {label}")
        plt.colorbar()

        plt.subplot(2, 3, 5)
        plot_image(C)
        plot_vf_grid(u_grad_rot["u_rot"], s=10, color="blue", scale=3.0)
        # vf_show(self.u_rot)
        plt.title(f"Orientation: {label}")

        plt.subplot(2, 3, 6)
        vf_show(u_grad_rot["u_rot"])
        plt.title(f"Orientation: {label}")
