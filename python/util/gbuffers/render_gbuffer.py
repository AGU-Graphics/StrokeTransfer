# -*- coding: utf-8 -*-
import os

import igl
from sklearn.preprocessing import MinMaxScaler

import util.gl.shaders.normal as normal_shader
import util.gl.shaders.picker as picker_shader
from util.canonical_sections.silhouette_orientation import \
    compute_silhouette_orientation_frame
from util.features.curvature_feature import (compute_gaussian_curvature,
                                             compute_mean_curvature)
from util.gbuffer import internal_file
from util.gl.renderer import *
from util.verbose import verbose_range


def load_scene_data(option, frame):
    json_file = option.camera_file_template.file(frame)
    blender_info = BlenderInfo(json_file)
    model_mat, view_mat, project_mat = blender_info.MVPMatrix()

    obj_file = option.object_file_template.file(frame)
    V, F = igl.read_triangle_mesh(obj_file)

    return V, F, model_mat, view_mat, project_mat


def render_color_gl(scalar_func, renderer, option, data_name, frame):
    out_path = internal_file(option, data_name, frame, dir_name="gbuffers")
    if os.path.exists(out_path) and not option.internal_overwrite:
        return

    V, F, model_mat, view_mat, project_mat = load_scene_data(option, frame)
    I = scalar_func(V, F)
    renderer.meshes = []

    I = np.clip(I, 0, 1)
    renderer.add_mesh(V, F, I)
    renderer.setMVPMat(model_mat, view_mat, project_mat)

    renderer.render(out_path)


def render_scalar(scalar_func, renderer, option, data_name, frame):
    out_path = internal_file(option, data_name, frame, "exr", dir_name="gbuffers")

    if os.path.exists(out_path) and not option.internal_overwrite:
        return

    V, F, model_mat, view_mat, project_mat = load_scene_data(option, frame)
    I = scalar_func(V, F)

    renderer.meshes = []
    scaler = MinMaxScaler()

    I = scaler.fit_transform(I.reshape(-1, 1)).flatten()

    renderer.add_mesh(V, F, I)
    renderer.setMVPMat(model_mat, view_mat, project_mat)

    O = renderer.render()

    h, w = O.shape[:2]

    O[:, :, :3] = scaler.inverse_transform(O[:, :, :3].reshape(-1, 3)).reshape(h, w, 3)
    imageio.imwrite(out_path, O)


def render_color(color_func, renderer, option, data_name, frame):
    render_color_gl(color_func, renderer, option, data_name, frame)


def render_orientation(orientation_func, renderer, option, data_name, frame):
    out_path = internal_file(option, data_name, frame, dir_name="gbuffers")
    if os.path.exists(out_path) and not option.internal_overwrite:
        return

    V, F, model_mat, view_mat, project_mat = load_scene_data(option, frame)
    u = orientation_func(V, F)

    renderer.meshes = []
    renderer.add_mesh(V, F, shader=normal_shader)
    renderer.meshes[0].set_orientation(u)
    renderer.setMVPMat(model_mat, view_mat, project_mat)

    renderer.render(out_file=out_path)


def render_normal():
    def func(renderer, option, frame):
        render_orientation(igl.per_vertex_normals, renderer, option, "Normal", frame)

    return func


def render_picker():
    data_name = "Picker"

    def func(renderer, option, frame):
        out_path = internal_file(option, data_name, frame, dir_name="gbuffers")
        if os.path.exists(out_path):
            return

        V, F, model_mat, view_mat, project_mat = load_scene_data(option, frame)
        renderer.meshes = []
        renderer.add_mesh(V, F, shader=picker_shader)
        renderer.setMVPMat(model_mat, view_mat, project_mat)

        renderer.render(out_file=out_path)

    return func


def render_position():
    def color_func(V, F):
        V_mean = np.mean(V, axis=0)
        V_center = V - V_mean
        V_min = np.min(V_center)
        V_max = np.max(V_center)

        C = (V_center - V_min) / (V_max - V_min)
        return C

    def func(renderer, option, frame):
        render_color(color_func, renderer, option, "Position", frame)

    return func


def compute_K(V, F):
    K = compute_gaussian_curvature(V, F)
    return K


def render_K():
    def func(renderer, option, frame):
        render_scalar(compute_K, renderer, option, "K", frame)

    return func


def compute_H(V, F):
    H = compute_mean_curvature(V, F)
    return H


def render_H():
    def func(renderer, option, frame):
        render_scalar(compute_H, renderer, option, "H", frame)

    return func


def out_gbuffer_frames(option, render_func, render_name, is_picker=False):
    img_file = option.diffuse_file_template.file(1)
    img = imageio.imread(img_file)
    renderer = Renderer(im_width=img.shape[1], im_height=img.shape[0], is_picker=is_picker)

    frames = option.frame_range()

    for frame in verbose_range(option.verbose, frames, desc=f"GBuffer-{render_name}"):
        render_func(renderer, option, frame)


def out_silhouette_orientation_frames(option):
    frames = option.frame_range()

    for frame in verbose_range(option.verbose, frames, desc=f"GBuffer-Silhouette Orientation"):
        compute_silhouette_orientation_frame(option, frame=frame)


def out_gbuffer(option):
    """ Save internal GBuffer data.

    Args:
        option: RegressionOption/TransferOption.
    """
    out_gbuffer_frames(option, render_K(), "K")
    out_gbuffer_frames(option, render_H(), "H")
    out_gbuffer_frames(option, render_position(), "Position")
    out_gbuffer_frames(option, render_normal(), "Normal")
    out_gbuffer_frames(option, render_picker(), "Picker", is_picker=True)
    out_silhouette_orientation_frames(option)
