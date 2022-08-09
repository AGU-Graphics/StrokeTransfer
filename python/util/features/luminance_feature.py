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
