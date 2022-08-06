import json

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline, Rbf
from sklearn.utils import shuffle


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


def load_annotation_file_old(json_file):
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
    for annotation in data:
        x = np.array(annotation["x"])
        y = np.array(annotation["y"])
        positions = np.array([x, y]).T
        width = annotation["width"]

        annotations.append(Annotation(positions, width))

    return annotations


def save_annotation_file(annotations, annotation_file):
    """ Save annotation data to the given file path.

    Args:
        annotations: list of annotation data (Annotation).
        annotation_file: output annotation data file (.json)
    """
    data = []

    for annotation in annotations:
        data_i = {}
        positions = annotation.positions
        data_i["x"] = positions[:, 0].tolist()
        data_i["y"] = positions[:, 1].tolist()
        data_i["width"] = annotation.width
        data.append(data_i)

    with open(annotation_file, 'w') as f:
        json.dump(data, f, indent=4)


class RBFModel:
    def __init__(self, k=1000000, smooth=1e-9):
        self.k = k
        self.smooth = smooth

    def fit(self, X, Y):
        XY = np.hstack([X, Y])
        if XY.shape[0] > self.k:
            samples = shuffle(XY, random_state=0)[:self.k]
            X_samples = samples[:, 0:X.shape[1]]
            Y_samples = samples[:, X.shape[1]:]
        else:
            X_samples = np.array(X)
            Y_samples = np.array(Y)

        rbfs = []

        for yi in range(Y_samples.shape[1]):
            Xs = []
            for xi in range(X_samples.shape[1]):
                Xs.append(X_samples[:, xi])
            Xs.append(Y_samples[:, yi])
            rbfi = Rbf(*Xs, smooth=self.smooth)
            rbfs.append(rbfi)

        self.rbfs = rbfs

    def transform(self, X):
        rbfs = self.rbfs
        Xs = []
        for xi in range(X.shape[1]):
            Xs.append(X[:, xi])
        Y = np.zeros((X.shape[0], len(rbfs)))

        for yi in range(Y.shape[1]):
            Y[:, yi] = rbfs[yi](*Xs)
        return Y


def normalize_vectors(u):
    epsilon = 1e-10
    u_norm = np.sqrt(np.sum(u * u, axis=1))
    return np.einsum("ij,i->ij", u, 1.0 / (epsilon + u_norm))


def grid_points(x_max, y_max, num_grids=20):
    xs = np.linspace(0, x_max, num_grids)
    ys = np.linspace(0, y_max, num_grids)

    X, Y = np.meshgrid(xs, ys)
    P = np.dstack((X, Y))
    return P


def arc_parameter(P):
    t = np.zeros((P.shape[0]))

    for i in range(P.shape[0] - 1):
        t[i + 1] = t[i] + np.linalg.norm(P[i + 1, :] - P[i, :])
    t /= t[-1]
    return t


def arc_length(P):
    t = np.zeros((P.shape[0]))

    for i in range(P.shape[0] - 1):
        t[i + 1] = t[i] + np.linalg.norm(P[i + 1, :] - P[i, :])
    return t[-1]


def curve_func(P):
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


def compute_X(P):
    X = [p for p in P]
    X = np.array(X)
    return X


def contour_directions(P0, dt=0.001):
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


def vf_constraints_from_annotations(annotations):
    V = []
    u = []
    L = []
    W = []

    for annotation in annotations:
        P = annotation.positions

        Wi = annotation.width

        V_, u_, L_ = contour_directions(P)

        V.extend(V_)
        u.extend(u_)
        L.extend([L_ for i in range(V_.shape[0])])
        W.extend([Wi for i in range(V_.shape[0])])

    V = np.array(V)

    u = np.array(u)
    L = np.array(L).reshape(-1, 1)
    W = np.array(W).reshape(-1, 1)

    return V, u, L, W


def interpolate_vector_field_from_annotations(annotations, x_max=1.0, y_max=1.0, num_grids=30):
    P, u, L, W = vf_constraints_from_annotations(annotations)

    X = compute_X(P)

    model = RBFModel(smooth=1e-9)

    Y = np.hstack((u, L, W))

    model.fit(X, Y)

    P = grid_points(x_max, y_max, num_grids)
    P = P.reshape(-1, 2)
    X = compute_X(P)

    Y = model.transform(X)
    u = normalize_vectors(Y[:, :2])

    P = X[:, :2]
    return P, u


def lines2annotations(lines):
    """ Wrapper data convert function from Line class to Annotation class.

    Args:
        lines: list of Line class instances.

    Returns:
        annotations: list of Annotation class instances.
    """
    annotations = [Annotation(positions=np.array(line.positions), width=line.width) for line in lines]
    return annotations


def interpolate_vector_field_from_gui(lines, x_max=1.0, y_max=1.0, num_grids=30):
    """ Interpolate orientations from the data exported by annotation tool.

    Args:
        lines: list of Line class instances (exported by annotation tool).
        x_max: maximum value of x coordinate (normalized by max(w, h))
        y_max: maximum value of y coordinate (normalized by max(w, h))
        num_grids: number of grids to interpolate orientations.

    Returns:
        P: (n, 2) np.array data for positions.
        u: (n, 2) np.array data for orientations.
    """
    annotations = lines2annotations(lines)
    interpolate_vector_field_from_annotations(annotations, x_max, y_max, num_grids)
    P, u = interpolate_vector_field_from_annotations(annotations, num_grids=30)
    return P,u
