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


import json
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

from util.fig_util import draw_bg, im_crop, plot_image, save_fig
from util.gbuffer import (internal_file, load_internal_orientation_frame)
from util.logger import getLogger
from util.normalize.norm import normalize_vectors
from util.rbf_model import RBFModel

logger = getLogger(__name__)


class Annotation:
    """ Data class for Annotation

    Attributes:
        position: (n, 2) np.array data for position list.
        width: float parameter value for the width.
    """

    def __init__(self, positions, width):
        """

        Args:
            positions: (n, 2) np.array data for position list.
            width: float parameter value for the width.
        """
        self.positions = positions
        self.width = width

    def __repr__(self):
        return f'Annotation(positions={self.positions}, width={self.width})'


class AnnotationSet:
    """ Interpolate annotated orientations, length, and width from the given annotation list.

    """

    def __init__(self, annotations, exemplar_img=None, N=None, A=None, version="1.0"):
        """

        Args:
            annotations: list of annotation data (Annotation).
            exemplar_img: (h, w, 4) exemplar image drawn by artist.
            N: (h, w, 4) normal image.
            A: (h, w) alpha mask image.
        """
        self.annotations = annotations

        self.exemplar_img = exemplar_img
        self.N = N

        if A is None:
            A = N[:, :, 3]
        self.A = A

        if exemplar_img is not None:
            self.im_shape = exemplar_img.shape

        self.model = vf_rbf_from_annotations(annotations)

        u, L, W = vf_image(self.model, self.exemplar_img, N)

        self.u = u
        self.L = L
        self.W = W
        self.A = A

    def orientation_image(self):
        """ Return the interpolated orientation image. """
        return self.u

    def stroke_length(self):
        """ Return the interpolated length image. """
        return self.L

    def stroke_width(self):
        """ Return the interpolated width image. """
        return self.W

    def exemplar_image(self):
        """ Return the exemplar image. """
        return self.exemplar_img

    def plot_exemplar_image(self):
        plot_image(self.exemplar_img)

    def plot_annoattions(self, width_scale=1.0):
        h, w = self.im_shape[:2]
        max_size = max(h, w)
        for annotation in self.annotations:
            P = annotation.positions

            width = max(annotation.width * max_size, 3.0)

            plt.plot(P[:, 0] * max_size, P[:, 1] * max_size, "o-", linewidth=0.5 * width * width_scale,
                     markersize=0.7 * width * width_scale)

    def plot_orientations(self, num_grids=40):
        h, w = self.im_shape[:2]
        N = self.N
        P, u = vf_on_grids(self.model, N, num_grids=num_grids)

        plt.quiver(P[:, 0], P[:, 1], u[:, 0], -u[:, 1], color=[0.05, 0.05, 0.05])


def image_points(width, height):
    xs = range(width)
    ys = range(height)
    X, Y = np.meshgrid(xs, ys)
    P = np.dstack((X, Y))
    return P


def grid_points(x_max, y_max, num_grids=20):
    xs = np.linspace(0, x_max, num_grids)
    ys = np.linspace(0, y_max, num_grids)

    X, Y = np.meshgrid(xs, ys)
    P = np.dstack((X, Y))
    return P


def normalize_vector(V):
    epsilon = 1e-10
    V_norm = epsilon + np.sqrt(np.einsum("ijk,ijk->ij", V, V))
    V_normalized = np.einsum("ijk, ij->ijk", V, 1.0 / V_norm)
    return V_normalized


def load_rgba(img_file):
    I = cv2.imread(img_file, -1)
    if I.dtype == np.uint16:
        I = np.float32(I) / 65535.0
    else:
        I = np.float32(I) / 255.0
    I = cv2.cvtColor(I, cv2.COLOR_BGRA2RGBA)
    return I


def load_mask(img_file):
    I = cv2.imread(img_file, -1)
    if I.dtype == np.uint16:
        I = np.float32(I) / 65535.0
    else:
        I = np.float32(I) / 255.0
    I = np.minimum(I[:, :, 0], I[:, :, 3])
    return I


def load_annotation_file_old(json_file):
    logger.debug(f"load_annotation_file_old: {os.getcwd()}")
    logger.debug(f"load_annotation_file_old: {os.path.abspath(json_file)}")

    with open(json_file, 'r') as f:
        json_data = f.read()
        data = json.loads(json_data)

    annotations = []
    for annotation in data:
        x = np.array(annotation["ix"])
        y = np.array(annotation["iy"])
        positions = np.array([x, y]).T
        width = annotation["w"] / 1024

        annotations.append(Annotation(positions, width))

    return annotations


def load_annotation_file(annotation_file):
    """ Load annotation data from the given file path.

    Args:
        annotation_file: input annotation data file (.json).

    Returns:
        annotations: list of annotation data (Annotation).
    """
    with open(annotation_file, 'r') as f:
        json_data = f.read()
        data = json.loads(json_data)

    annotations = []
    version = "1.0"
    for i, annotation in enumerate(data):
        if i == 0:
            if "v" in annotation.keys():
                version = annotation["v"]
        x = np.array(annotation["x"])
        y = np.array(annotation["y"])
        positions = np.array([x, y]).T
        width = annotation["width"]

        annotations.append(Annotation(positions, width))

    logger.debug(f"load_annotation_file: version={version}")

    return annotations, version


def save_annotation_file(annotations, annotation_file):
    """ Save annotation data to the given file path.

    Args:
        annotations: list of annotation data (Annotation).
        annotation_file: output annotation data file (.json)
    """
    data = []

    for i, annotation in enumerate(annotations):
        data_i = {}
        positions = annotation.positions
        data_i["x"] = positions[:, 0].tolist()
        data_i["y"] = positions[:, 1].tolist()
        data_i["width"] = annotation.width

        if i == 0:
            data_i["v"] = "0.0"
        data.append(data_i)

    with open(annotation_file, 'w') as f:
        json.dump(data, f, indent=4)


def load_annotation(option):
    """ Load annotation data with the given option.

    Args:
        option: input setting data (AnnotationOption).

    Returns:
        annnotation_set: interpolated orientations, length, width data (AnnotationSet).
    """
    N = load_internal_orientation_frame(option, "Normal", frame=option.frame_start, scale=1, format="png",
                                        dir_name="gbuffers")
    A = N[:, :, 3]

    json_file = option.annotation_filename
    style_file = option.exemplar_filename

    style_img = load_rgba(style_file)

    annotations, version = load_annotation_file(json_file)
    return AnnotationSet(annotations, style_img, N, A, version)


def arc_parameter(P):
    t = np.zeros((P.shape[0], 1))

    for i in range(P.shape[0] - 1):
        t[i + 1] = t[i] + np.linalg.norm(P[i + 1, :] - P[i, :])
    t /= t[-1]
    return t


def arc_length(P):
    t = np.zeros((P.shape[0], 1))

    for i in range(P.shape[0] - 1):
        t[i + 1] = t[i] + np.linalg.norm(P[i + 1, :] - P[i, :])
    return t[-1]


def curve_func(P, version="1.0"):
    if version == "0.0":
        k = int(min(P.shape[1] - 1, 3))
    else:
        k = int(min(P.shape[0] - 1, 3))

    t = arc_parameter(P)

    x = P[:, 0].flatten()
    y = P[:, 1].flatten()

    fx = InterpolatedUnivariateSpline(t, x, k=k)
    fy = InterpolatedUnivariateSpline(t, y, k=k)

    def func(t_new):
        x = fx(t_new)
        y = fy(t_new)
        return np.dstack((x, y)).reshape(-1, 2)

    return func, t


def contour_directions(P0, version="1.0", dt=0.001, sp=0.05):
    if len(P0) < 2:
        return [], [], []

    t = arc_parameter(P0)

    f, t = curve_func(P0)

    try:
        f, t = curve_func(P0)
    except:
        print(f"Error: {len(P0)}")
        return [], [], []

    num_samples = 2 * P0.shape[0]

    t = np.linspace(0.0, 1.0, num_samples)
    P = f(t)
    L = arc_length(P)
    t1 = np.clip(t + dt, 0, 1)
    t0 = np.clip(t - dt, 0, 1)
    u = f(t1) - f(t0)

    u = normalize_vectors(u)
    return P, u, L


def vf_constraints_from_annotations(annotations, version="1.0"):
    V = []
    u = []
    L = []
    W = []

    for annotation in annotations:
        P = annotation.positions

        Wi = annotation.width

        V_, u_, L_ = contour_directions(P, version)

        V.extend(V_)
        u.extend(u_)
        L.extend([L_ for i in range(V_.shape[0])])
        W.extend([Wi for i in range(V_.shape[0])])

    V = np.array(V)

    u = np.array(u)
    L = np.array(L).reshape(-1, 1)
    W = np.array(W).reshape(-1, 1)

    return V, u, L, W


def compute_X(P):
    X = [p for p in P]
    X = np.array(X)
    return X


def vf_rbf_from_annotations(annotations, version="1.0"):
    V, u, L, W = vf_constraints_from_annotations(annotations, version)

    X = compute_X(V)

    rbf_model = RBFModel(k=1000, smooth=1e-9)

    Y = np.hstack((u, L, W))

    rbf_model.fit(X, Y)

    return rbf_model


def vf_on_grids(model, N, x_max=1.0, y_max=1.0, num_grids=20):
    P = grid_points(x_max, y_max, num_grids)
    P = P.reshape(-1, 2)

    X = compute_X(P)

    Y = model.transform(X)
    u = normalize_vectors(Y[:, :2])

    h, w = N.shape[:2]
    P = X[:, :2]

    max_size = max(h, w)

    P[:, 0] *= max_size - 1
    P[:, 1] *= max_size - 1

    u[:, 0] *= max_size - 1
    u[:, 1] *= max_size - 1

    return P, u


class ImageScaler:
    def __init__(self, scale=0.25):
        self.scale = scale

    def transform(self, X):
        scale = self.scale
        h, w, cs = X.shape
        self.X_shape = X.shape
        h_low = int(h * scale)
        w_low = int(w * scale)

        X_low = np.zeros((h_low, w_low, cs))
        for ci in range(cs):
            X_low[:, :, ci] = cv2.resize(X[:, :, ci], (w_low, h_low))
        return X_low

    def inverse_transform(self, X_low):
        X_shape = self.X_shape
        h, w = X_shape[:2]
        cs = X_low.shape[2]
        X = np.zeros((h, w, cs))

        for ci in range(cs):
            X[:, :, ci] = cv2.resize(X_low[:, :, ci], (w, h))
        return X


def vf_image(model, I, N=None):
    A = N[:, :, 3]
    h, w = I.shape[:2]

    P = np.float32(image_points(w, h))
    P = P.reshape(-1, 2)

    max_size = max(w, h)

    P[:, 0] /= max_size - 1
    P[:, 1] /= max_size - 1

    u_dash = np.zeros_like(I)

    X = compute_X(P)

    scaler = ImageScaler(scale=0.5)
    X_ = X.reshape(h, w, -1)
    X_ = scaler.transform(X_)

    h_low, w_low = X_.shape[:2]
    Y_ = model.transform(X_.reshape(h_low * w_low, -1))
    Y = scaler.inverse_transform(Y_.reshape(h_low, w_low, -1))
    Y = Y.reshape(h * w, -1)

    u_dash_flat = Y[:, :2]
    u_dash_flat = normalize_vectors(u_dash_flat)

    u_dash[:, :, :2] = u_dash_flat.reshape(u_dash.shape[0], u_dash.shape[1], -1)

    L = Y[:, 2].reshape(h, w)
    W = Y[:, 3].reshape(h, w)

    L = np.clip(L, 3.0 / max_size, 10.0)
    W = np.clip(W, 3.0 / max_size, 10.0)

    if A is None:
        A = I[:, :, 3]

    for ci in range(3):
        u_dash[:, :, ci] *= A

    u_dash[:, :, 3] = A
    return u_dash, L, W


def save_annotation_plot(option, xlim=(0.0, 1.0), ylim=(0.0, 1.0),
                         is_white=True, with_vf=False, out_file=None):
    import seaborn as sns
    sns.set()

    annotation_set = load_annotation(option)

    fig_size = 10
    area_size = min(xlim[1] - xlim[0], ylim[1] - ylim[0])
    width_scale = fig_size / 10.0 / area_size
    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = plt.subplot(1, 1, 1)
    annotation_set.plot_exemplar_image()
    if with_vf:
        annotation_set.plot_orientations()
    if is_white:
        draw_bg(0.5 * np.ones_like(annotation_set.A), [1.0, 1.0, 1.0])
    annotation_set.plot_annoattions(width_scale=width_scale)
    im_crop(ax, annotation_set.A, xlim, ylim)

    out_file = internal_file(option, data_name="annotation_plot", frame=option.frame_start, format="png")

    save_fig(out_file)
    plt.close(fig)


def plot_annotations(annotations):
    for annotation in annotations:
        P = annotation.positions
        Q, u = contour_directions(P)
        if len(Q) == 0:
            continue
        plt.quiver(Q[:, 0], Q[:, 1], u[:, 0], -u[:, 1])


if __name__ == '__main__':
    annotations = load_annotation_file_old("../annotation/style1.json")
    save_annotation_file(annotations, "../annotation/annotation.json")
