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
from scipy.interpolate import Rbf
from sklearn.utils import shuffle


class RBFModel:
    def __init__(self, k=1000000, smooth=1e-3):
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
