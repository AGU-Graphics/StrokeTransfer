# -*- coding: utf-8 -*-
import cv2
import igl
import imageio
import json
import numpy as np
import os
import point_cloud_utils as pcu

from glumpy import app, gl, gloo

from util.blender_info import BlenderInfo
from util.gl.mesh import GLMesh
from util.integral_curve import IntegralCurveData, integral_curve_order, integral_curve_random
from util.stroke_polygon_generator import stroke_polygon_generator, vertex_project
from util.vf_2d_to_3d import unproject_scalar_by_image_sampling
from util.verbose import verbose_print, verbose_range

app.use("glfw")

# Create glumpy app window with the specified width and height.
win_width, win_height = 1024, 1024
window = app.Window(width=win_width, height=win_height, color=(0.2, 0.2, 0.2, 1))


class ImageData:
    """
        Args:
            option: StrokeOption.
            frame: input frame number.

        Attributes:
            color_img: (im_height, im_width, 4) RGBA color image data.
            img_height: image height.
            img_width: image width.
    """

    def __init__(self, option, frame):
        self.color_img = cv2.cvtColor(cv2.imread(option.color_file_template % frame, -1), cv2.COLOR_BGRA2RGBA).astype(
            np.float32) / 255.
        self.img_height = self.color_img.shape[0]
        self.img_width = self.color_img.shape[1]


class MeshData:
    """ Mesh data class

    Attributes:
        V: (#V, 3) vertices.
        F: (#F, 3) face indices.
        omega: discrete 1-form data (on edges).
        EV: (#E, 2) list of edges described as pair of vertices.
        FE: (#F, 3) list storing triangle-edge relation.
        EF: (#E, 2) list storing edge-triangle relation, uses -1 to indicate boundaries.
        FE_flip: (#F, 3) flipped flags to make counter-clockwise edge loop.
        areas: (#F, ) areas on faces.
        N_f: (#F, 3) per face normals.
        N_e: (#E, 3) per edge normals.
        length: (#V, ) stroke length data on vertices.
        width: (#V, ) stroke width data on vertices.
    """

    def __init__(self, V, F, omega):
        """
        Args:
            V: (#V, 3) vertices.
            F: (#F, 3) face indices.
            omega: discrete 1-form data (on edges).
        """
        self.V = V
        self.F = F
        self.omega = omega

        self.EV, self.FE, self.EF = igl.edge_topology(V, F)

        FE = self.FE.reshape(-1, 3)

        FE0 = self.EV[FE, 0]
        FE_flip = np.ones_like(FE)
        FE_flip[FE0 != F] = -1
        self.FE_flip = FE_flip

        self.areas = igl.doublearea(V, F)
        self.N_f = igl.per_face_normals(V, F, np.array([0.0, 0.0, 1.0]))

        N_e, _, _ = igl.per_edge_normals(V, F, 0, self.N_f)
        N_e_n = np.sqrt(np.einsum("ij,ij->i", N_e, N_e))
        for i in range(N_e.shape[1]):
            N_e[:, i] = N_e[:, i] / N_e_n
        self.N_e = N_e

        self.length = -1
        self.width = -1

    def create_length(self, option, frame, camera_info):
        model_mat, view_mat, project_mat = camera_info.MVPMatrix()
        if option.length_multiplier:
            length_img = imageio.imread(option.length_file_template % frame).astype(np.float64)
            length = unproject_scalar_by_image_sampling(length_img, self.V, model_mat, view_mat, project_mat).reshape(
                -1)
            self.length = length * option.length_multiplier + option.length_offset
        else:
            self.length = np.ones(self.V.shape[0]) * option.length_offset
        verbose_print(option.verbose, f"<WidthImage> min: {np.min(self.length)}, max: {np.max(self.length)}")

    def create_width(self, option, frame, camera_info):
        model_mat, view_mat, project_mat = camera_info.MVPMatrix()
        if option.width_multiplier:
            width_img = imageio.imread(option.width_file_template % frame).astype(np.float64)
            width = unproject_scalar_by_image_sampling(width_img, self.V, model_mat, view_mat, project_mat).reshape(-1)
            self.width = width * option.width_multiplier + option.width_offset
        else:
            self.width = np.ones(self.V.shape[0]) * option.width_offset
        verbose_print(option.verbose, f"<WidthImage> min: {np.min(self.width)}, max: {np.max(self.width)}")


class AnchorMeshData:
    """ Mesh data for anchor points.

    Attributes:
        level: current level.
        num: counts of anchor points at the current level.
        F_idx: face indices where anchor points are assigned.
        barycentric_coordinate: barycentric coordinates that determine the location of anchor points.
        point: anchor point coordinates in object space.
        point_proj:  anchor point coordinates in screen space.
        length: length data on anchor points.
        width: width data on anchor points.
        vector_angle: stroke angle variation on anchor points.
        flag: active flags of each anchor point.
        img_x: x pixel coordinates of anchor points.
        img_y: y pixel coordinates of anchor points.
        color: colors on anchor points.
    """

    def __init__(self, level, option, anchor_data, mesh_data, image_data, camera_info):
        """
        Args:
            level: current level.
            option: StrokeOption.
            anchor_data: anchor point data.
            mesh_data: mesh data.
            image_data: image data.
            camera_info: camera info exported from blender.
        """
        anchor_nums = anchor_data["anchor_point_nums"]
        all_anchor_F_idx = np.array(anchor_data["F_idx"])
        all_anchor_barycentric_coordinate = np.array(anchor_data["barycentric_coordinate"])

        self.level = level

        if level == 0:
            self.num = anchor_nums[option.num_levels - 1]
            self.F_idx = all_anchor_F_idx
            self.barycentric_coordinate = all_anchor_barycentric_coordinate
        else:
            self.num = anchor_nums[level - 1]
            self.F_idx = all_anchor_F_idx[:self.num]
            self.barycentric_coordinate = all_anchor_barycentric_coordinate[:self.num]

        self.point = pcu.interpolate_barycentric_coords(mesh_data.F, self.F_idx, self.barycentric_coordinate,
                                                        mesh_data.V)
        self.point_proj = vertex_project(camera_info, self.point)

        self.length = self.interpolation_scalar(mesh_data.F, mesh_data.length)
        self.width = self.interpolation_scalar(mesh_data.F, mesh_data.width)
        self.vector_angle = -1
        self.flag = -1

        self.img_x = (self.point_proj[:, 0] * image_data.img_width).astype(np.int)
        self.img_y = ((1 - self.point_proj[:, 1]) * image_data.img_height).astype(np.int)

        self.color = image_data.color_img[np.clip(self.img_y, 0, image_data.img_height),
                     np.clip(self.img_x, 0, image_data.img_width), :]

    def create_vector_angle(self, angular_file_name):
        with open(angular_file_name) as f:
            angular_data = json.load(f)
        all_vector_angle = np.array(angular_data["angular"])
        self.vector_angle = all_vector_angle[:self.num]

    def interpolation_scalar(self, F, S):
        SF = S[F[self.F_idx, :]]
        D = np.einsum("ij, ij->i", SF, self.barycentric_coordinate)
        return D


def add_integral_curve_datas(take_integral_curve_datas, give_integral_curve_datas):
    """ Append integral curve data for the given datas.

    Args:
        take_integral_curve_datas: original integral curve data to be extended.
        give_integral_curve_datas: new integral curve data to append.

    Returns:
        take_integral_curve_datas: appended take_integral_curve_datas (overwritten in this function).
    """

    take_integral_curve_datas.lines.extend(give_integral_curve_datas.lines)
    take_integral_curve_datas.lines_normals.extend(give_integral_curve_datas.lines_normals)
    take_integral_curve_datas.colors.extend(give_integral_curve_datas.colors)
    take_integral_curve_datas.widths.extend(give_integral_curve_datas.widths)
    take_integral_curve_datas.indexes.extend(give_integral_curve_datas.indexes)
    return take_integral_curve_datas


def add_integral_curve_datas_sort(take_integral_curve_datas, give_integral_curve_datas):
    """ Append integral curve data (sorted) for the given datas.

    Args:
        take_integral_curve_datas: original integral curve data to be extended.
        give_integral_curve_datas: new integral curve data to append (data will be sorted by luminance).

    Returns:
        take_integral_curve_datas: appended take_integral_curve_datas (overwritten in this function).
    """

    if len(give_integral_curve_datas.colors) != 0:
        give_colors_data = np.array(give_integral_curve_datas.colors).reshape(-1, 1, 4).astype(np.float32)
        give_colors_lab_data = cv2.cvtColor(give_colors_data[:, :, :3], cv2.COLOR_RGB2Lab)
        l_data = give_colors_lab_data[:, :, 0].reshape(-1)
        sort_idx = np.argsort(l_data, kind="marge")

        take_integral_curve_datas.lines.extend([give_integral_curve_datas.lines[sort_id] for sort_id in sort_idx])
        take_integral_curve_datas.lines_normals.extend(
            [give_integral_curve_datas.lines_normals[sort_id] for sort_id in sort_idx])
        take_integral_curve_datas.colors.extend([give_integral_curve_datas.colors[sort_id] for sort_id in sort_idx])
        take_integral_curve_datas.widths.extend([give_integral_curve_datas.widths[sort_id] for sort_id in sort_idx])
        take_integral_curve_datas.indexes.extend([give_integral_curve_datas.indexes[sort_id] for sort_id in sort_idx])
    return take_integral_curve_datas


def create_angular_file(option, angular_file_name, num):
    """ Save angular data in the target_file
    
    Args:
        option: StrokeOption.
        angular_file_name: file name to store angular offsets.
        num: target number of generated anchor points.
    """
    verbose_print(option.verbose, "save angular file.")
    # np.random.seed(10)
    if option.max_random_angular_offset:
        all_vector_angle = ((np.random.rand(
            num)) * 2 * option.max_random_angular_offset) - option.max_random_angular_offset
    else:
        all_vector_angle = np.zeros(num)
    os.makedirs(os.path.dirname(angular_file_name), exist_ok=True)
    angular_data = {"angular": list(all_vector_angle)}
    with open(angular_file_name, 'w') as f:
        json.dump(angular_data, f, indent=4)


def rendering_stroke(option, frame):
    """ Generate integral curves and render them as stroke drawing.

    Args:
        option: StrokeOption.
        frame: input frame number.
    """

    """ Load file data. """
    image_data = ImageData(option, frame)

    texture = np.zeros((image_data.img_height, image_data.img_width, 4), np.float32).view(gloo.TextureFloat2D)
    depth_buffer = gloo.DepthBuffer(image_data.img_width, image_data.img_height, format=gl.GL_DEPTH_COMPONENT)
    framebuffer = gloo.FrameBuffer(color=[texture], depth=depth_buffer)

    previous_frame_use = False if option.frame_start == frame or not option.coherence else True

    # model, view projection matrix from blender information.
    camera_info = BlenderInfo(option.camera_file_template % frame)
    model_mat, view_mat, project_mat = camera_info.MVPMatrix()

    # mesh data (vertices, faces) from obj file.
    V, F = igl.read_triangle_mesh(option.object_file_template % frame)

    # load discrete 1-form data for the input orientations.
    with open(option.orientation_file_template % frame) as f:
        vector_data = json.load(f)
    # omega: discrete 1-form data (on edges).
    omega = np.array(vector_data["orientation"])

    anchor_file_name = option.anchor_file_template % option.frame_start if option.coherence else option.anchor_file_template % frame
    with open(anchor_file_name) as f:
        anchor_data = json.load(f)
    anchor_nums = anchor_data["anchor_point_nums"]

    angular_file_name = f"{option.internal}/angular.json"
    if not os.path.exists(angular_file_name) or not option.coherence or option.internal_overwrite:
        create_angular_file(option, angular_file_name, anchor_nums[-1])

    def draw_strokes(strokes_data):
        """ Render strokes of the given stroke data.

        Args:
            strokes_data: stroke data for drawing.

        Returns:
            rendering_image: stroke rendering output image.
        """

        window.clear()
        framebuffer.activate()
        gl.glViewport(0, 0, framebuffer.width, framebuffer.height)

        gl.glClearColor(0, 0, 0, 0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_BACK)

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glPolygonOffset(8.0, 1.0)

        # object-stroke depth test is enabled to hide the strokes occluded by the object.
        object_model.draw()

        # stroke-stroke depth test is disabled.
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glDepthMask(gl.GL_FALSE)

        # stroke render loop.
        for stroke in verbose_range(option.verbose, strokes_data):
            stroke.draw()

        # enable ordinary depth test function.
        gl.glDepthMask(gl.GL_TRUE)
        framebuffer.deactivate()
        rendering_image = np.flipud(framebuffer.color[0].get())
        return rendering_image

    def create_object_model():
        """ Generate base model data used in rendering process.
        Returns:
            model: GLMesh mesh instance with the normalized data.
        """

        # vertex positions are adjusted along view vector (aim for robust depth test).
        view_location = camera_info.camera.world[:, 3]
        view_location = np.array([view_location[0], view_location[2], -view_location[1]])
        view_vector = V - view_location
        view_vector_norm = view_vector / np.tile(np.sqrt(
            view_vector[:, 0] * view_vector[:, 0] + view_vector[:, 1] * view_vector[:, 1] + view_vector[:,
                                                                                            2] * view_vector[:, 2]),
                                                 (3, 1)).T

        bbox = np.max(V, axis=0) - np.min(V, axis=0)
        move_V_offset = np.min(bbox) * 0.03

        move_V = V + move_V_offset * view_vector_norm
        model = GLMesh(move_V, F, np.zeros(F.shape) + 0.5)
        model.setModelMatrix(model_mat)
        model.setViewMatrix(view_mat)
        model.setProjectionMatrix(project_mat)
        return model

    """ stroke settings. """
    # stroke data settings.
    half_strokes = []
    strokes = []
    all_integral_curve_datas = IntegralCurveData()
    current_frame_integral_curve_datas = IntegralCurveData()

    object_model = create_object_model()

    mesh_data = MeshData(V, F, omega)
    mesh_data.create_length(option, frame, camera_info)
    mesh_data.create_width(option, frame, camera_info)

    half_stroke_image = draw_strokes(half_strokes)

    max_stroke_num = 0
    anchor_flag_level_zero = -1
    previous_frame_index = -1
    for level in range(option.num_levels + 1):
        if level == 0 and not previous_frame_use:
            continue
        else:
            """ Create anchor data. """
            anchor_mesh_data = AnchorMeshData(level, option, anchor_data, mesh_data, image_data, camera_info)
            anchor_mesh_data.create_vector_angle(angular_file_name)

            if level == 0:
                previous_frame_index_file = f"{option.internal}/index/index_{frame - 1:03}.json"
                with open(previous_frame_index_file) as f:
                    previous_frame_index_data = json.load(f)
                previous_frame_index = np.array(previous_frame_index_data["index"])

                anchor_flag_level_zero = np.ones(anchor_mesh_data.num)
                if previous_frame_index.shape[0] > 0:
                    anchor_flag_level_zero[previous_frame_index] = 0
                max_stroke_num = previous_frame_index.shape[0]
            else:
                anchor_mesh_data.flag = np.ones(
                    anchor_mesh_data.num) if not previous_frame_use else anchor_flag_level_zero[:anchor_mesh_data.num]
                anchor_mesh_data.flag[0:max_stroke_num] = 0
                max_stroke_num = anchor_mesh_data.num

        """ Create integral curve. """
        verbose_print(option.verbose, f"Create Integral Curves: level {level} ({max_stroke_num}).")
        if level == 0 and previous_frame_use:
            current_level_integral_curve_datas = \
                integral_curve_order(image_data, mesh_data, anchor_mesh_data, max_stroke_num, camera_info,
                                     previous_frame_index, option.verbose)
            all_integral_curve_datas = add_integral_curve_datas(all_integral_curve_datas,
                                                                current_level_integral_curve_datas)
        else:
            current_level_integral_curve_datas = \
                integral_curve_random(image_data, mesh_data, anchor_mesh_data, max_stroke_num, camera_info,
                                      half_stroke_image, option.verbose)
            current_frame_integral_curve_datas = add_integral_curve_datas(current_frame_integral_curve_datas,
                                                                          current_level_integral_curve_datas)

        """ Create stroke polygon. """
        if level < option.num_levels and (
                not len(all_integral_curve_datas.lines) + len(current_frame_integral_curve_datas.lines) == anchor_nums[
                    option.num_levels - 1]):
            verbose_print(option.verbose, f"Create Strokes: level {level}.")
            half_strokes = stroke_polygon_generator(current_level_integral_curve_datas, image_data, half_strokes, 0,
                                                    camera_info, option.texture_file, option.verbose, True)
            verbose_print(option.verbose, "Rendering texture.")
            half_stroke_image = draw_strokes(half_strokes)
        verbose_print(option.verbose,
                      f"Number of strokes in level {level}: {len(all_integral_curve_datas.lines) + len(current_frame_integral_curve_datas.lines)}")
        verbose_print(option.verbose, "")

    if option.coherence and not option.drawing_incremental:
        all_integral_curve_datas = add_integral_curve_datas_sort(all_integral_curve_datas,
                                                                 current_frame_integral_curve_datas)
    else:
        all_integral_curve_datas = add_integral_curve_datas(all_integral_curve_datas,
                                                            current_frame_integral_curve_datas)

    index_file_name = f"{option.internal}/index/index_{frame:03}.json"
    os.makedirs(os.path.dirname(index_file_name), exist_ok=True)
    index_data = {"index": all_integral_curve_datas.indexes}
    with open(index_file_name, 'w') as f:
        json.dump(index_data, f, indent=4)

    """ Rendering stroke. """
    if option.drawing_incremental and option.drawing_start_frame > frame:
        stroke_image = np.flipud(framebuffer.color[0].get())
    else:
        drawing_stroke_num = option.drawing_num_incremental_strokes_per_frame * (
                    frame - option.drawing_start_frame + 1) if option.drawing_incremental else 0
        verbose_print(option.verbose, "Create Strokes.")
        strokes = stroke_polygon_generator(all_integral_curve_datas, image_data, strokes, drawing_stroke_num,
                                           camera_info, option.texture_file, option.verbose, False)
        verbose_print(option.verbose, f"Number of strokes: {len(strokes)}")
        verbose_print(option.verbose, "")
        verbose_print(option.verbose, "Rendering Stroke.")
        stroke_image = draw_strokes(strokes)

    os.makedirs(os.path.dirname(option.output_stroke_template % frame), exist_ok=True)
    if option.output_stroke_template % frame is not None:
        imageio.imwrite(option.output_stroke_template % frame, np.uint8(255 * stroke_image))

    app.run(framecount=0)
    app.quit()
    verbose_print(option.verbose, "")
