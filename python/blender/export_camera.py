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


import argparse
import json
import os
import sys

import bpy
import numpy as np


def blendermat2python(A):
    """ Convert blender matrix data to python array matrix.
    Args:
        A: blender matrix data (mathutils.Matrix).

    Returns:
        A_python: pure python array matrix.
    """
    A_python = []

    for i in range(4):
        A_row = []
        for j in range(4):
            A_row.append(A[i][j])
        A_python.append(A_row)
    return A_python


def export_camera(frame_start=1, frame_end=5, use_scene_frame_range=False):
    """ export camera info from blender.

    Args:
        frame_start: first frame number.
        frame_end: last frame number.
        use_scene_frame_range: if True, frame_start and frame_end will be set from belnder scene setting.
    """
    if use_scene_frame_range:
        frame_start = bpy.context.scene.frame_start
        frame_end = bpy.context.scene.frame_end

    scene_dir = os.path.dirname(bpy.data.filepath)

    for frame in range(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(frame)

        # obtain active camera
        scene = bpy.context.scene
        camera = scene.camera
        camera_data = camera.data
        p_camera = camera.location
        R_camera2world = camera.matrix_world

        camera_json = {}
        camera_json["location"] = [p_camera.x, p_camera.y, p_camera.z]
        camera_json["world"] = blendermat2python(R_camera2world)

        scene_render = bpy.context.scene.render
        sensor_height = scene_render.resolution_y / scene_render.resolution_x * camera_data.sensor_width

        camera_json["flocal_length"] = camera_data.lens
        camera_json["film_x"] = camera_data.sensor_width
        camera_json["film_y"] = sensor_height

        camera_json["angle_x"] = camera_data.angle_x
        camera_json["angle_y"] = camera_data.angle_y

        camera_json["near"] = camera_data.clip_start
        camera_json["far"] = camera_data.clip_end

        fovy = camera_data.angle_x if camera_data.sensor_width >= sensor_height else camera_data.angle_y
        f = 1 / np.tan(fovy / 2)
        aspect = camera_data.sensor_width / sensor_height
        near = camera_data.clip_start
        far = camera_data.clip_end
        p_mat = [[f, 0, 0, 0], [0, f * aspect, 0, 0],
                 [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)], [0, 0, -1, 0]]

        camera_json["project_mat"] = p_mat

        # output.
        data_json = {}
        data_json["camera"] = camera_json

        filepath = os.path.join(scene_dir, 'camera/camera_{:03}.json'.format(frame))
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with  open(filepath, 'w') as f:
            json.dump(data_json, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_start', type=int, default=1)
    parser.add_argument('--frame_end', type=int, default=5)
    parser.add_argument('--use_scene_frame_range', action='store_true')

    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    export_camera(args.frame_start, args.frame_end, args.use_scene_frame_range)
