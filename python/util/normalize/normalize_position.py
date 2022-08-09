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
