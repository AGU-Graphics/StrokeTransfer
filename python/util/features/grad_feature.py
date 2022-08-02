import cv2
import numpy as np
from cv2.ximgproc import guidedFilter

from util.blender_info import BlenderInfo
from util.features.luminance_feature import lumiance_feature
from util.fig_util import *
from util.gbuffer import (load_frame_by_name, load_internal_frame_by_name,
                          load_internal_orientation_frame)
from util.logger import getLogger
from util.normalize.norm import normalize_vector_image

logger = getLogger(__name__)


def IQR_clamp(X, A, q=1):
    x = X[A > 0.5]

    q1 = q
    q2 = 100 - q

    q2, q1 = np.percentile(x, [q2, q1])

    y = np.array(X)
    y[y < q1] = q1
    y[q2 < y] = q2
    return y


def grad_norm(I, film_x, N=None):
    """ Compute gradient feature for the given image.

    Args:
        I: input image for gradient operation.
        film_x: film size in x-direction of 3D camera infromation.
        N: (h, w, 4) normal image.

    Returns:
        nabla_I: (h, w) gradient feature image.
    """
    logger.debug(f"N: {N.shape}")
    logger.debug(f"I: {I.shape}")
    if N is not None:
        I = guidedFilter(N[:, :, :3], np.float32(I), 3, 1e-9)
    gx = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=1)
    gy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=1)

    h, w = I.shape[:2]
    r = np.sqrt(w ** 2 + h ** 2)

    d = film_x / w

    nabla_I = np.sqrt(gx ** 2 + gy ** 2) / d

    return nabla_I


def grad_dir(L, I):
    """ Compute gradient direction for the given luminance.
    
    Args:
        L: (h, w) luminance image.
        I: (h, w, 4) input image with alpha channel.

    Returns:
        G: (h, w, 4) gradient orientation image.
    """
    G = np.zeros_like(I)

    L = cv2.GaussianBlur(L, (0, 0), sigmaX=10.0)
    gx = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=1)
    gy = cv2.Sobel(L, cv2.CV_64F, 0, 1, ksize=1)

    G[:, :, 0] = gx
    G[:, :, 1] = gy

    G = normalize_vector_image(G)

    for ci in range(3):
        G[:, :, ci] *= I[:, :, 3]
    G[:, :, 3] = I[:, :, 3]
    return G

def luminance(I):
    return luminance_gray(I)


def luminance_Lab(I):
    Lab = cv2.cvtColor(I[:, :, :3], cv2.COLOR_RGB2Lab)
    return Lab[:, :, 0] / 100.0


def luminance_gray(I):
    return np.einsum("ijk,k->ij", I[:, :, :3], np.array([0.2126, 0.7152, 0.0722]))


def change_luminance(I, L_out):
    return change_luminance_Lab(I, L_out)


def change_luminance_Lab(I, L_out):
    Lab = cv2.cvtColor(I[:, :, :3], cv2.COLOR_RGB2Lab)
    Lab[:, :, 0] = 100.0 * L_out
    I_tone = np.array(I)
    I_tone[:, :, :3] = cv2.cvtColor(Lab, cv2.COLOR_Lab2RGB)
    return I_tone


def change_lumiannce_gray(I, L_out):
    epsilon = 1e-6
    L_in = luminance(I)
    return np.einsum("ijk,ij->ijk", I, L_out / (L_in + epsilon))


def tone_mapping_sigmoid(L, k=1.39, c=1.67):
    return k * (2.0 * np.exp(c * L) / (np.exp(c * L) + 1.0) - 1.0)


def tone_mapping_color(I):
    L = luminance_Lab(I)
    L_out = tone_mapping_sigmoid(L)
    return change_luminance_Lab(I, L_out)


def load_grad(option, I_file_template, frame=1, scale=1.0):
    """ Load gradient feature for the given image file template.

    Args:
        option: RegressionOption/TransferOption.
        I_file_template: image file template used for gradient operation.
        frame: target frame.
        scale: scale parameter for image size.

    Returns:
        nabla_I: (h, w) gradient feature image.
    """
    json_file = option.camera_file_template.file(frame)
    blender_info = BlenderInfo(json_file)
    film_x = blender_info.camera.film_x

    I = load_frame_by_name(I_file_template, frame=frame, scale=scale)

    N = load_internal_orientation_frame(option, "Normal", frame=frame, scale=scale, format="png", dir_name="gbuffers")

    L = lumiance_feature(I)

    nabla_I = grad_norm(L, film_x, N)
    nabla_I = tone_mapping_sigmoid(nabla_I)

    return nabla_I
