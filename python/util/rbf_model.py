# -*- coding: utf-8 -*-
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
