import os
import pickle
from random import gauss

import cv2
import matplotlib.cm as cm
import numpy as np
from matplotlib import pyplot as plt

from util.annotation import load_annotation, save_annotation_plot
from util.canonical_sections.canonical_sections import \
    compute_canonical_sections
from util.features.features import compute_features
from util.fig_util import draw_bg, im_crop, plot_image, plot_vf_grid, save_fig
from util.gbuffer import load_internal_orientation_frame
from util.logger import getLogger
from util.model.scalar_model import NearestNeighborModel
from util.model.vf_model import VectorFieldRegressionModel, proj_vf
from util.normalize.norm import normalize_vector_image

logger = getLogger(__name__)


def internal_file(option, data_name, frame=None, format="png", dir_name=None):
    if dir_name is None:
        out_dir = f"{option.internal}/{data_name}"
    else:
        out_dir = f"{option.internal}/{dir_name}"
    os.makedirs(out_dir, exist_ok=True)
    if frame is None:
        out_path = f"{out_dir}/{data_name}.{format}"
    else:
        out_path = f"{out_dir}/{data_name}_{frame:03d}.{format}"
    return out_path


def constraints_vf(option, annotation_set):
    """

    Args:
        option: RegressionOption/TransferOption.
        annotation_set: target annotation data.

    Returns:
        AW_constraints: constraints for A_hat W_u matrix.
        u_constraints: constraints for the target orientation.
    """
    if option.model_type_orientation == "1st-order":
        vf_model = VectorFieldRegressionModel(order=1)
    elif option.model_type_orientation == "2nd-order":
        vf_model = VectorFieldRegressionModel(order=2)
    elif option.model_type_orientation == "0th-order":
        vf_model = VectorFieldRegressionModel(order=0)
    else:
        vf_model = VectorFieldRegressionModel(order=1)

    scale = 0.5

    u_dash = annotation_set.orientation_image()

    canonical_sections = compute_canonical_sections(option, frame=option.frame_start, scale=scale)
    features = compute_features(option, frame=option.frame_start, scale=scale)

    for key in canonical_sections.canonical_sections.keys():
        vf_model.set_orientation(key, canonical_sections.canonical_sections[key])

    for key in features.features.keys():
        vf_model.set_feature(key, features.features[key])

    N = load_internal_orientation_frame(option, "Normal", frame=option.frame_start, scale=scale, format="png", dir_name="gbuffers")
    N[:, :, 1] *= -1.0

    h, w = N.shape[:2]
    u_dash = cv2.resize(u_dash, (w, h))
    u_dash = proj_vf(u_dash, N)

    A = annotation_set.A
    A = cv2.resize(A, (w, h))

    AW_constraints, u_constraints = vf_model.constraints(u_dash, A)
    return vf_model, AW_constraints, u_constraints


def fit_vf_constraints(vf_model, AW_constraints, u_constraints):
    vf_model.fit_constraints(AW_constraints, u_constraints)
    return vf_model


def out_model_matrix(option, model):
    if model.order != 1:
        return
    model_matrix = model.model_matrix()
    weight_max = np.max(np.abs(model_matrix))

    fig = plt.figure(figsize=(16, 10))
    ax = plt.subplot(1, 1, 1)
    plot_image(model_matrix, cmap=cm.PuOr, vmin=-weight_max, vmax=weight_max)
    out_file = internal_file(option, data_name="model_matrix")
    save_fig(out_file)


def set_canonical_sections_and_features(option, vf_model, scale=1.0):
    """ Set canonical sections and features for the given vector field model.

    Args:
        option:  RegressionOption/TransferOption.
        vf_model: target vector field model.
        scale: scale parameter for image size.
    """
    canonical_sections = compute_canonical_sections(option, frame=option.frame_start, scale=scale)
    features = compute_features(option, frame=option.frame_start, scale=scale)

    for key in canonical_sections.canonical_sections.keys():
        vf_model.set_orientation(key, canonical_sections.canonical_sections[key])

    for key in features.features.keys():
        vf_model.set_feature(key, features.features[key])

    return vf_model


def fit_vf_model(option, annotation_set):
    """ Fit vector field model for the given annotation set.

    Args:
        option: RegressionOption.
        annotation_set: target annotation data.

    Returns:
        vf_model: Fitted vector field model.
    """

    if option.model_type_orientation == "1st-order":
        vf_model = VectorFieldRegressionModel(order=1)
    elif option.model_type_orientation == "2nd-order":
        vf_model = VectorFieldRegressionModel(order=2)
    elif option.model_type_orientation == "0th-order":
        vf_model = VectorFieldRegressionModel(order=0)
    else:
        vf_model = VectorFieldRegressionModel(order=1)

    u_dash = annotation_set.orientation_image()

    scale = 0.5

    vf_model = set_canonical_sections_and_features(option, vf_model, scale)

    N = load_internal_orientation_frame(option, "Normal", frame=option.frame_start, scale=scale, format="png", dir_name="gbuffers")
    N[:, :, 1] *= -1.0

    h, w = N.shape[:2]
    u_dash = cv2.resize(u_dash, (w, h))
    u_dash = proj_vf(u_dash, N)

    A = annotation_set.A
    A = cv2.resize(A, (w, h))

    vf_model.fit(u_dash, A)
    out_model_matrix(option, vf_model)
    vf_model.clean_internal()

    return vf_model


def target_width(annotation_set):
    """ Return the target width image from the annotation data.

    Args:
        annotation_set: target annotation data.

    Returns:
        W: (h, w) target width image from the annotation data.
    """
    return annotation_set.stroke_width()


def target_length(annotation_set):
    """ Return the target length image from the annotation data.

        Args:
            annotation_set: target annotation data.

        Returns:
            L: (h, w) target length image from the annotation data.
    """
    return annotation_set.stroke_length()


def target_color(annotation_set):
    """ Return the target color image from the annotation data.

        Args:
            annotation_set: target annotation data.

        Returns:
            C: (h, w, 4) target color image from the annotation data.
    """
    return annotation_set.exemplar_image()


def constraints_clw(option, annotation_set, target_func):
    """ Return the constrains for the given target from the annotation data.

    Args:
        option: RegressionOption/RegressionMultiOption.
        annotation_set: target annotation data.
        target_func: target_width/target_length/target_color.

    Returns:
        W_u_constraints: constraints for feature matrix.
        I_constraints: constraints for the target color/length/width.
    """
    scale = 1.0
    features = compute_features(option, frame=option.frame_start, scale=scale)

    model = NearestNeighborModel()

    h, w = features.feature_shape
    for key in features.features.keys():
        model.set_feature(key, features.features[key])

    I = target_func(annotation_set)

    I = cv2.resize(I, (w, h))
    A = annotation_set.A
    A = cv2.resize(A, (w, h))

    W_u_constraints, I_constraints = model.constraints(I, A)
    return W_u_constraints, I_constraints


def fit_clw_constraints(W_u_constraints, I_constraints):
    """ Fit color/length/width model.

    Args:
        W_u_constraints: constraints for feature matrix.
        I_constraints: constraints for the target color/length/width.

    Returns:
        model: fitted color/length/width field model.
    """
    model = NearestNeighborModel()
    model.fit_constraints(W_u_constraints, I_constraints)
    model.clean_internal()

    return model


def fit_clw_model(option, annotation_set, target_func):
    """ Fit color/length/width model.

    Args:
        option: RegressionOption/RegressionMultiOption.
        annotation_set: target annotation data.
        target_func: target_color/target_length/target_width.

    Returns:
        model: fitted color/length/width field model.
    """
    scale = 1.0
    features = compute_features(option, frame=option.frame_start, scale=scale)

    model = NearestNeighborModel()

    h, w = features.feature_shape
    for key in features.features.keys():
        model.set_feature(key, features.features[key])

    I = target_func(annotation_set)

    I = cv2.resize(I, (w, h))
    A = annotation_set.A
    A = cv2.resize(A, (w, h))

    model.fit(I, A)
    model.clean_internal()

    return model


def fit_color_model(option, annotation_set):
    """ Fit color model.

    Args:
        option: RegressionOption/RegressionMultiOption.
        annotation_set: target annotation data.

    Returns:
        model: fitted color field model.
    """
    return fit_clw_model(option, annotation_set, target_color)


def fit_length_model(option, annotation_set):
    """ Fit length model.

    Args:
        option: RegressionOption/RegressionMultiOption.
        annotation_set: target annotation data.

    Returns:
        model: fitted length field model.
    """
    return fit_clw_model(option, annotation_set, target_length)


def fit_width_model(option, annotation_set):
    """ Fit width model.

    Args:
        option: RegressionOption/RegressionMultiOption.
        annotation_set: target annotation data.

    Returns:
        model: fitted width field model.
    """
    return fit_clw_model(option, annotation_set, target_width)


def load_model(model_file):
    """ Load vector/color/length/width model from the given pickle file.

    Args:
        model_file: pickle model file.

    Returns:
        model: loaded vector/color/length/width model.
    """
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    return model


def save_model(model, model_file):
    """ Save vector/color/length/width model to the given pickle file.

    Args:
        model: vector/color/length/width model.
        model_file: pickle model file.
    """
    model_dir = os.path.dirname(model_file)
    os.makedirs(model_dir, exist_ok=True)
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)


def out_models(option):
    """ Save the models for the regression output settings.

    Args:
        option: RegressionOption.
    """
    annotation_set = load_annotation(option)
    save_annotation_plot(option)

    vf_model = fit_vf_model(option, annotation_set)
    save_model(vf_model, option.model_output_orientation)

    out_regression(option, vf_model, annotation_set)

    color_model = fit_color_model(option, annotation_set)
    save_model(color_model, option.model_output_color)

    length_model = fit_length_model(option, annotation_set)
    save_model(length_model, option.model_output_length)

    width_model = fit_width_model(option, annotation_set)
    save_model(width_model, option.model_output_width)


def out_models_multi(option):
    """ Save the models for the regression with multiple exemplars.

    Args:
        option: RegressionMultiOption.
    """
    out_vf_model_multi(option)
    out_clw_models_multi(option, target_color, option.model_output_color)
    out_clw_models_multi(option, target_length, option.model_output_length)
    out_clw_models_multi(option, target_width, option.model_output_width)


def out_vf_model_multi(option):
    """ Save the vector field model for the regression with multiple exemplars.

    Args:
        option: RegressionMultiOption.

    Note:
        Use model.constraints() -> model.fit_constraints framework to handle multiple exemplars.
    """
    AW_constraints = []
    u_constraints = []
    for regression_option in option.regression_settings:
        annotation_set = load_annotation(regression_option)
        vf_model, AW_samples, u_samples = constraints_vf(regression_option, annotation_set)

        logger.debug(f"AW_samples.shape: {AW_samples.shape}")
        logger.debug(f"u_samples.shape: {u_samples.shape}")

        AW_constraints.append(AW_samples)
        u_constraints.append(u_samples)

    AW_constraints = np.vstack(AW_constraints)
    u_constraints = np.hstack(u_constraints)

    logger.debug(f"AW_constraints.shape: {AW_constraints.shape}")
    logger.debug(f"u_constraints.shape: {u_constraints.shape}")

    vf_model = fit_vf_constraints(vf_model, AW_constraints, u_constraints)
    out_model_matrix(option, vf_model)
    save_model(vf_model, option.model_output_orientation)

    for regression_option in option.regression_settings:
        annotation_set = load_annotation(regression_option)
        out_regression(regression_option, vf_model, annotation_set)


def out_clw_models_multi(option, target_func, model_output):
    """ Save the color/length/width field model for the regression with multiple exemplars.

    Args:
        option: RegressionMultiOption.

    Note:
        Use model.constraints() -> model.fit_constraints framework to handle multiple exemplars.
    """
    W_u_constraints = []
    I_constraints = []
    for regression_option in option.regression_settings:
        annotation_set = load_annotation(regression_option)

        W_u_samples, I_samples = constraints_clw(regression_option, annotation_set, target_func)

        W_u_constraints.append(W_u_samples)
        I_constraints.append(I_samples)

    W_u_constraints = np.array(W_u_constraints)
    W_u_constraints = W_u_constraints.reshape(-1, W_u_constraints.shape[2])

    I_constraints = np.array(I_constraints)
    I_constraints = I_constraints.reshape(-1, I_constraints.shape[2])

    logger.debug(f"W_u_constraints.shape: {W_u_constraints.shape}")
    logger.debug(f"I_constraints.shape: {I_constraints.shape}")

    model = fit_clw_constraints(W_u_constraints, I_constraints)
    save_model(model, model_output)


def regression_result(option, annotation_set, vf_model):
    """ Compute regression result for the given annotation data and vector field model.

    Args:
        option: RegressionOption/RegressionMultiOption.
        annotation_set: the target annotation data.
        vf_model: the target vector field model.

    Returns:
        u_dash: (h, w, d) target orientation image.
        u_tilde: (h, w, d) fitted orientation image.
        N: (h, w, 4) normal image.
    """
    scale = 0.5
    u_dash = annotation_set.orientation_image()

    N = load_internal_orientation_frame(option, "Normal", frame=option.frame_start, scale=scale, format="png", dir_name="gbuffers")
    N[:, :, 1] *= -1.0

    h, w = N.shape[:2]

    u_dash = cv2.resize(u_dash, (w, h))
    u_dash[:, :, :3] = normalize_vector_image(u_dash[:, :, :3])
    u_dash = proj_vf(u_dash, N)

    set_canonical_sections_and_features(option, vf_model, scale=scale)

    u_tilde = vf_model.predict()
    u_tilde = proj_vf(u_tilde, N)
    u_tilde[:, :, 3] = N[:, :, 3]
    return u_dash, u_tilde, N


def out_regression(option, vf_model, annotation_set):
    """ Save the output regression result.

    Args:
        option: RegressionOption/RegressionMultiOption.
        vf_model: the target vector field model.
        annotation_set: the target annotation data.
    """
    u_dash, u_tilde, N = regression_result(option, annotation_set, vf_model)
    out_regression_direction_plot(option, u_dash, u_tilde, N)
    out_regression_evaluation(option, u_dash, u_tilde, N)


def out_regression_direction_plot(option, u_dash, u_tilde, N):
    """ Save the regression result to compare the fitted orientations with the input orientations.

    Args:
        option: RegressionOption/RegressionMultiOption.
        u_dash: (h, w, d) target orientation image.
        u_tilde: (h, w, d) fitted orientation image.
        N: (h, w, 4) normal image.
    """

    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1)
    plot_image(N[:, :, 2], cmap=cm.bone, vmin=-0.1, vmax=0.9)
    draw_bg(N[:, :, 3], bg_color=[0.8, 0.8, 0.8])

    scale = 5.0
    s = 15

    plot_vf_grid(u_dash, s=s, color=np.array([0.0, 0.0, 1.0]), scale=scale)
    plot_vf_grid(u_tilde, s=s, color=np.array([1.0, 0.0, 0.0]), scale=scale)

    xlim = [0.05, 0.85]
    ylim = [0.15, 0.8]
    im_crop(ax, N, xlim, ylim)

    out_file = internal_file(option, data_name="regression_orientation", frame=option.frame_start,
                             dir_name="regression")
    save_fig(out_file)


def gen_random_dir(dim=3):
    """ Generate random direction vector.

    Args:
        dim: size of the target vector data.

    Returns:
        v: random direction with the specified size of the target vector data.
    """
    v = np.array([gauss(0, 1) for i in range(dim)])
    v /= np.linalg.norm(v)
    if v[2] < 0.0:
        v *= -1
    return v


def plot_fit_evaluation(u_dash, u_tilde, ax):
    """ Plot the regression result to compare the fitted orientations with the input orientations.

    Args:
        u_dash: (h, w, d) target orientation image.
        u_tilde: (h, w, d) fitted orientation image.
        ax: maplot figure axis.
    """
    q_dash = []
    q_tilde = []

    dim = u_tilde.shape[1]

    num_dirs = 1000

    for i in range(num_dirs):
        fit_dir = gen_random_dir(dim)

        rand_ids = np.random.randint(u_dash.shape[0], size=1000)

        u_dash_sample = u_dash[rand_ids, :]
        u_fit_sample = u_tilde[rand_ids, :]

        q_dash_i = np.einsum("j,ij->i", fit_dir, u_dash_sample)
        q_tilde_i = np.einsum("j,ij->i", fit_dir, u_fit_sample)

        q_dash.extend(q_dash_i)
        q_tilde.extend(q_tilde_i)

    v_min = np.min([np.min(q_dash), np.min(q_tilde)])
    v_max = np.max([np.max(q_dash), np.max(q_tilde)])

    edges = np.linspace(-1.0, 1.0, 80)

    plt.hist2d(q_dash, q_tilde, bins=(edges, edges), cmap=cm.magma, density=True)
    plt.plot((v_min, v_max), (v_min, v_max), "--", c=[0.8, 0.8, 0.8])
    plt.axis("off")
    # plt.xticks([-1, 0, 1])
    # plt.yticks([-1, 0, 1])
    # ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    # ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))

    ax.set_aspect('equal')


def out_regression_evaluation(option, u_dash, u_tilde, N):
    """ Save the alignment plot for the fitted model.

    Args:
        option: RegressionOption.
        u_dash: (h, w, d) target orientation image.
        u_tilde: (h, w, d) fitted orientation image.
        N: (h, w, 4) normal image.
    """
    alpha_region = N[:, :, 3] > 0.5
    u_fit_sel = u_tilde[alpha_region, :3]
    u_dash_sel = u_dash[alpha_region, :3]

    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1)
    plot_fit_evaluation(u_dash_sel, u_fit_sel, ax)

    out_file = internal_file(option, data_name="regression_plot", frame=option.frame_start, dir_name="regression")
    save_fig(out_file)
