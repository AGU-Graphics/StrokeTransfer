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
import numpy as np

from util.logger import getLogger
from util.normalize.norm import normalize_vector_image

logger = getLogger(__name__)

import imageio


def load_rgba(img_file):
    I = cv2.imread(img_file, -1)
    if I.dtype == np.uint16:
        I = np.float32(I) / 65535.0
    else:
        I = np.float32(I) / 255.0
    I = cv2.cvtColor(I, cv2.COLOR_BGRA2RGBA)
    return I


def load_exr(img_file):
    I = imageio.imread(img_file)

    logger.debug(f"load_exr: I.shape={I.shape}")

    if I.ndim == 2:
        return I

    A = I[:, :, 3]

    if not np.all(A) <= 0.5:
        I_min = np.min(I[A > 0.5, :3])
        I_max = np.max(I[A > 0.5, :3])

        I[:, :, :3] = np.clip(I[:, :, :3], I_min, I_max)

    float_max = 1e20
    I = np.clip(I, -float_max, float_max)
    return I


def save_rgba(file_path, I):
    I_8u = np.array(I)
    if I_8u.dtype != np.uint8:
        I_8u = np.uint8(255 * I_8u)
    I_8u = cv2.cvtColor(I_8u, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(file_path, I_8u)


def save_exr(file_path, I):
    imageio.imwrite(file_path, np.float32(I))


def save_npz(file_path, I):
    np.savez_compressed(file_path, a=I)


def load_npz(img_file):
    loaded = np.load(img_file)
    return loaded['a']


def load_frame_by_name(file_template, frame=1, scale=1):
    img_file = file_template.file(frame)
    format = file_template.file_extension

    if not os.path.exists(img_file):
        return None
    if "png" == format:
        I = load_rgba(img_file)
    elif "exr" == format:
        I = load_exr(img_file)
    elif "npz" == format:
        I = load_npz(img_file)

    I = cv2.resize(I, None, fx=scale, fy=scale)
    return I


def load_orientation_frame(file_template, frame=1, scale=1):
    N = load_frame_by_name(file_template, frame, scale)
    if N is None:
        return None

    N[:, :, :3] = 2 * N[:, :, :3] - 1.0

    return N


def save_frame_by_name(I, out_template, frame=1, format="png"):
    out_file = out_template % frame
    out_dir = os.path.dirname(out_file)
    os.makedirs(out_dir, exist_ok=True)

    logger.debug(f"save_frame_by_name: {out_file}... {I.shape}")

    I_ = I

    if "png" == format:
        save_rgba(out_file, I_)
    elif "exr" == format:
        save_exr(out_file, I_)
    elif "npz" == format:
        save_npz(out_file, I_)


def internal_file(option, data_name, frame, format="png", dir_name=None):
    out_dir = f"{option.internal}/{data_name}"
    if dir_name is not None:
        out_dir = f"{option.internal}/{dir_name}/{data_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{data_name}_{frame:03d}.{format}"
    return out_path


def load_internal_frame_by_name(option, data_name, frame=1, scale=1, format="png", dir_name=None):
    img_file = internal_file(option, data_name, frame, format, dir_name=dir_name)

    if not os.path.exists(img_file):
        return None
    if "png" == format:
        I = load_rgba(img_file)
    elif "exr" == format:
        I = load_exr(img_file)
    elif "npz" == format:
        I = load_npz(img_file)

    I = cv2.resize(I, None, fx=scale, fy=scale)

    logger.debug(f"load_internal_frame_by_name: {img_file}... {I.shape}")

    return I

def dilate_boundary(I, A):
    I_smooth = np.array(I)
    sigma = 3
    for iter in range(5):
        I_smooth = cv2.GaussianBlur(I_smooth, (0, 0), sigma)

        I_smooth = (1.0 - A) * I_smooth + A * I
    return I_smooth

def pre_orientation(N):
    A = N[:,:,3]
    for ci in range(3):
        N[:,:,ci] = A * N[:,:,ci]

    A = np.einsum("ijk,ijk->ij", N[:,:,:3], N[:,:,:3])

    sigma = 3

    N_smooth = np.array(N)
    for iter in range(5):
        N_smooth = cv2.GaussianBlur(N_smooth, (0, 0), sigma)

        for ci in range(3):
            N_smooth[:, :, ci] = (1.0 - A) * N_smooth[:, :, ci] + A * N[:, :, ci]

    N_smooth[:, :, :3] = normalize_vector_image(N_smooth[:, :, :3])
    N_smooth[:, :, 3] = N[:, :, 3]

    return N_smooth

def load_internal_orientation_frame(option, data_name, frame=1, scale=1, format="png", dir_name=None):
    N = load_internal_frame_by_name(option, data_name, frame, scale=scale, format=format, dir_name=dir_name)
    if N is None:
        return None

    N[:, :, :3] = 2 * N[:, :, :3] - 1.0
    N = pre_orientation(N)

    return N


def save_internal_frame_by_name(I, option, data_name, frame=1, scale=1, format="png", dir_name=None):
    out_file = internal_file(option, data_name, frame, format, dir_name=dir_name)

    out_dir = os.path.dirname(out_file)
    os.makedirs(out_dir, exist_ok=True)
    I_ = I

    if "png" == format:
        save_rgba(out_file, I_)
    elif "exr" == format:
        save_exr(out_file, I_)
    elif "npz" == format:
        save_npz(out_file, I_)
