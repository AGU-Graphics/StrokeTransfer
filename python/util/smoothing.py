import json
import os

import igl
import numpy as np
import tqdm
from pydec import laplace_derham, simplicial_complex
from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import dsolve

from util.logger import getLogger
from util.verbose import verbose_range

logger = getLogger(__name__)


class VectorFieldSmoothing:
    """ Smoothing optimization for vector field. """

    def __init__(self, L, star1, dt=1.0 / 24.0):
        """

        Args:
            L: (#E, #E) discrete covariant Laplacian matrix for 1-form.
            star1: (#E, #E) discrete Hodge star matrix for 1-form.
            dt: time interval.
        """
        self.L = L
        self.star1 = star1
        self.dt = dt

    def smoothing_spatial(self, c_e, lambda_spatial):
        """ Smoothing operator for spatial filtering.

        Args:
            c_e: (#E,) target discrete 1-form.
            lambda_spatial: regularization parameter for spatial coherence.

        Returns:
            c_e_tilde: (#E,) optimized discrete 1-form.
        """
        L = self.L
        star1 = self.star1

        A = csr_matrix(star1 + lambda_spatial * L)
        b = star1 * c_e

        c_e_tilde = dsolve.spsolve(A, b)
        return c_e_tilde

    def smoothing(self, c_e, c_e_tilde_pre, lambda_spatial, lambda_temporal):
        """ Smoothing operator for spatial and temporal filtering.

        Args:
            c_e: (#E,) target discrete 1-form on current frame.
            c_e_tilde_pre: (#E,) optimized discrete 1-form on previous frame.
            lambda_spatial: regularization parameter for spatial coherence.
            lambda_temporal: regularization parameter for temporal coherence.

        Returns:
            c_e_tilde: (#E,) optimized discrete 1-form.
        """
        c_e_tilde_spatial = self.smoothing_spatial(c_e, lambda_spatial)

        star1 = self.star1
        dt = self.dt

        A = csr_matrix((1.0 + lambda_temporal / dt) * star1)
        b = star1 * (c_e_tilde_spatial + (lambda_temporal / dt) * c_e_tilde_pre)

        c_e_tilde = dsolve.spsolve(A, b)
        return c_e_tilde


def load_model_data(object_file_template, frame):
    """ Load Obj model data.

    Args:
        object_file_template: filename template for obj file.
        frame: target frame number.

    Returns:
        V: (#V, 3) vertices.
        F: (#F, 3) face indices.
    """
    obj_file = object_file_template.file(frame)
    V, F = igl.read_triangle_mesh(obj_file)
    return V, F


def save_Laplacian_mat(temp_dir, object_file_template, frame=1):
    """ Save discrete covariant Laplacian matrix.

    Args:
        temp_dir: directory for internal data on smoothing process.
        object_file_template: filename template for obj file.
        frame: target frame number.
    """
    V, F = load_model_data(object_file_template, frame)

    # PyDEC for covariant Laplacian matrix.
    sc = simplicial_complex((V, F))
    L = laplace_derham(sc.get_cochain_basis(1)).v

    data_name = "Laplacian"
    mat_file = f"{temp_dir}/smoothing/{data_name}/{data_name}_{frame:03}.mtx"
    out_dir = os.path.dirname(mat_file)

    os.makedirs(out_dir, exist_ok=True)

    mmwrite(mat_file, L)


def load_Laplacian_mat(temp_dir, object_file_template, frame=1, update=False):
    """ Load discrete covariant Laplacian matrix.

    Args:
        temp_dir: directory for internal data on smoothing process.
        object_file_template: filename template for obj file.
        frame: target frame number.
        update: False => skip the generation of laplacian matrix data if the target file exists.

    Returns:
        L: (#E, #E) discrete covariant Laplacian matrix for 1-form.
    """
    data_name = "Laplacian"
    mat_file = f"{temp_dir}/smoothing/{data_name}/{data_name}_{frame:03}.mtx"

    need_update = (not os.path.exists(mat_file)) or update

    if need_update:
        save_Laplacian_mat(temp_dir, object_file_template, frame)
    L = mmread(mat_file)
    logger.debug(f"Laplacian: mean={L.mean()}")
    return L


def save_HodgeStar1_mat(temp_dir, object_file_template, frame=1):
    """ Save discrete Hodge star matrix for 1-form.

        Args:
            temp_dir: directory for internal data on smoothing process.
            object_file_template: filename template for obj file.
            frame: target frame number.
    """

    V, F = load_model_data(object_file_template, frame)

    # PyDEC for covariant Laplacian matrix.
    sc = simplicial_complex((V, F))
    star1 = sc[1].star

    data_name = "HodgeStar1"
    mat_file = f"{temp_dir}/smoothing/{data_name}/{data_name}_{frame:03}.mtx"
    out_dir = os.path.dirname(mat_file)

    os.makedirs(out_dir, exist_ok=True)

    mmwrite(mat_file, star1)


def load_HodgeStar1_mat(temp_dir, object_file_template, frame=1, update=False):
    """ Load discrete covariant Laplacian matrix.

        Args:
            temp_dir: directory for internal data on smoothing process.
            object_file_template: filename template for obj file.
            frame: target frame number.
            update: False => skip the generation of laplacian matrix data if the target file exists.

        Returns:
            star1: (#E, #E) discrete Hodge star matrix for 1-form.
        """

    data_name = "HodgeStar1"

    mat_file = f"{temp_dir}/smoothing/{data_name}/{data_name}_{frame:03}.mtx"

    need_update = (not os.path.exists(mat_file)) or update

    if need_update:
        save_HodgeStar1_mat(temp_dir, object_file_template, frame)
    star1 = mmread(mat_file)
    logger.debug(f"HodgeStar1: mean={star1.mean()}")

    return star1


def load_orientation(file_template, frame=1):
    """ Load orientation for the target frame.

    Args:
        file_template: file name template for the target orientation.
        frame: target frame number.

    Returns:
        c_e: (#E, ) discrete 1-form for edge-based orientation field.
    """
    in_file = file_template % frame

    if not os.path.exists(in_file):
        return None

    with open(in_file) as fp:
        json_data = json.load(fp)

    c_e = np.array(json_data["orientation"])
    return c_e


def save_orientation(file_template, c_e, frame=1):
    """ Save orientation for the target frame.

    Args:
        file_template: file name template for the target orientation.
        c_e: (#E, ) discrete 1-form for edge-based orientation field.
        frame: target frame number.
    """
    out_c_e_file = file_template % frame

    out_dir = os.path.dirname(out_c_e_file)

    os.makedirs(out_dir, exist_ok=True)

    with open(out_c_e_file, 'w') as fp:
        json.dump({"orientation": c_e.tolist()}, fp, indent=4)


def out_smoothing_frame(vf_smoothing, option, frame):
    """ Save smoothing data on the target frame.

    Args:
        vf_smoothing: instance of VectorFieldSmoothing.
        option: parameter settings for VectorFieldSmoothing.
        frame: target frame number.

    """
    c_e = load_orientation(option.orientation_file_template, frame)

    c_e_pre = load_orientation(option.output_file_template, frame - 1)

    if c_e_pre is None:
        c_e_tilde = vf_smoothing.smoothing_spatial(c_e, option.lambda_spatial)
    else:
        c_e_tilde = vf_smoothing.smoothing(c_e, c_e_pre, option.lambda_spatial, option.lambda_temporal)

    save_orientation(option.output_file_template, c_e_tilde, frame)


def out_smoothing_static(option):
    """ Save smoothing result for static object.

    Args:
        option: parameter settings for VectorFieldSmoothing.

    """

    # load matrix elements.
    L = load_Laplacian_mat(option.internal, option.object_file_template, frame=1, update=option.internal_overwrite)
    star1 = load_HodgeStar1_mat(option.internal, option.object_file_template, frame=1, update=option.internal_overwrite)

    vf_smoothing = VectorFieldSmoothing(L, star1)

    # smoothing for target frames.
    frames = option.frame_range()

    for frame in verbose_range(option.verbose, frames, desc=f"Vector Field Smoothing"):
        out_smoothing_frame(vf_smoothing, option, frame)


def out_smoothing_deform(option):
    """ Save smoothing result for deforming object.

    Args:
        option: parameter settings for VectorFieldSmoothing.

    """
    frames = option.frame_range()

    # smoothing for target frames.
    for frame in verbose_range(option.verbose, frames, desc=f"Vector Field Smoothing"):
        # load matrix elements on each frame.
        L = load_Laplacian_mat(option.internal, option.object_file_template, frame=frame, update=option.internal_overwrite)
        star1 = load_HodgeStar1_mat(option.internal, option.object_file_template, frame=frame,
                                    update=option.internal_overwrite)

        vf_smoothing = VectorFieldSmoothing(L, star1)
        out_smoothing_frame(vf_smoothing, option, frame)


def out_smoothing(option):
    """ Save smoothing result for static/deforming object.

    Args:
        option: parameter settings for VectorFieldSmoothing.

    """
    if option.deform:
        out_smoothing_deform(option)
    else:
        out_smoothing_static(option)
