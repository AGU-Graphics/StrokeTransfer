import numpy as np


def normalize_point01(p, w, h):
    max_size = max(w, h)
    q = np.array(p)
    q[0] *= w / max_size
    q[1] *= h / max_size
    return q


def normalize_point02(p, w, h):
    r = np.sqrt(w ** 2 + h ** 2)
    q = np.array(p)
    q[0] *= w / r
    q[1] *= h / r
    return q


def normalize_positions01(P, w, h):
    max_size = max(w, h)
    Q = np.array(P)

    Q[:, 0] *= w / max_size
    Q[:, 1] *= h / max_size
    return Q


def normalize_positions02(P, w, h):
    r = np.sqrt(w ** 2 + h ** 2)
    Q = np.array(P)

    Q[:, 0] *= w / r
    Q[:, 1] *= h / r
    return Q


def inverse_normalize_positions01(P, w, h):
    return P


def normalize_length(l, w, h):
    r = np.sqrt(w ** 2 + h ** 2)
    return l / r
