import igl
import numpy as np

from util.blender_info import BlenderInfo
from util.gbuffer import load_internal_frame_by_name


def IQR_clamp_image(X, A, q=1):
    x = X[A > 0.5]

    q1 = q
    q2 = 100 - q

    q2, q1 = np.percentile(x, [q2, q1])

    y = np.array(X)
    y[y < q1] = q1
    y[q2 < y] = q2
    return y


def IQR_clamp(x, q=0.5):
    q1 = q
    q2 = 100 - q
    q2, q1 = np.percentile(x, [q2, q1])

    y = np.array(x)
    y[y < q1] = q1
    y[q2 < y] = q2
    return y


def IQR_q_abs(x, q=1):
    q1 = q
    x_abs = np.abs(x)
    q2 = 100 - q1
    q2, q1 = np.percentile(x_abs, [q2, q1])
    return q2


def load_scene_data(scene_data, frame):
    json_file = scene_data.file_path('data/Status/Status_{:03}.json'.format(frame))
    blender_info = BlenderInfo(json_file)
    model_mat, view_mat, project_mat = blender_info.MVPMatrix()

    obj_file = scene_data.file_path('data/Model/Model_{:03}.obj'.format(frame))
    V, F = igl.read_triangle_mesh(obj_file)

    return V, F, model_mat, view_mat, project_mat


def load_curvature(option, ch_name="K", frame=1, scale=1.0, is_debug=False):
    K = load_internal_frame_by_name(option, ch_name, frame=frame, scale=scale, format="exr", dir_name="gbuffers")
    K = K[:, :, 0]

    return K


def standard_vertices(V):
    bb_size = np.mean(np.max(V, axis=0) - np.min(V, axis=0))
    return V / bb_size


def principal_curvature(V, F):
    return igl.principal_curvature(standard_vertices(V), F)


def face_areas(V, F):
    areas = igl.doublearea(standard_vertices(V), F) / 2
    return areas


def vertex_areas(V, F):
    area_mat = igl.massmatrix(standard_vertices(V), F)
    return area_mat.diagonal()


def compute_gaussian_curvature(V, F):
    T1, T2, k1, k2 = principal_curvature(V, F)
    K = k1 * k2
    K = IQR_clamp(K)
    return K


def compute_mean_curvature(V, F):
    T1, T2, k1, k2 = principal_curvature(V, F)
    H = (k1 + k2) / 2.0
    H = IQR_clamp(H)
    return H
