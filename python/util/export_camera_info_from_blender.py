import argparse
import json
import os
import sys

import bpy
import numpy as np


def blendermat2python(A):
    A_python = []

    for i in range(4):
        A_row = []
        for j in range(4):
            A_row.append(A[i][j])
        A_python.append(A_row)
    return A_python


def main(frame_start, frame_end, use_scene_frame_range):
    if use_scene_frame_range:
        nStart = bpy.context.scene.frame_start
        nEnd = bpy.context.scene.frame_end
    else:
        nStart = frame_start
        nEnd = frame_end

    scene_dir = os.path.dirname(bpy.data.filepath)

    for frame in range(nStart, nEnd + 1):
        bpy.context.scene.frame_set(frame)

        # camera_data
        camera = bpy.data.objects['Camera']
        camera_data = bpy.data.objects['Camera'].data
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

        # output_part
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
    main(args.frame_start, args.frame_end, args.use_scene_frame_range)
