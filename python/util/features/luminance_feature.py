import cv2
import numpy as np


def luminance(I):
    return luminance_gray(I)


def luminance_Lab(I):
    Lab = cv2.cvtColor(I[:, :, :3], cv2.COLOR_RGB2Lab)
    return Lab[:, :, 0] / 100.0


def luminance_gray(I):
    return np.einsum("ijk,k->ij", I[:, :, :3], np.array([0.2126, 0.7152, 0.0722]))


def tone_mapping_sigmoid(L, k=1.39, c=1.67):
    return k * (2.0 * np.exp(c * L) / (np.exp(c * L) + 1.0) - 1.0)


def lumiance_feature(I):
    L = luminance(I)
    L = tone_mapping_sigmoid(L)
    return L
