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
import os
import sys

import bpy


def export_object(frame_start, frame_end, use_scene_frame_range):
    """ export 3D object data (.obj) from blender.

    Note: only export visible objects (skip hidden objects).

    Args:
        frame_start: first frame number.
        frame_end: last frame number.
        use_scene_frame_range: if True, frame_start and frame_end will be set from belnder scene setting.
    """

    if use_scene_frame_range:
        frame_start = bpy.context.scene.frame_start
        frame_end = bpy.context.scene.frame_end

    scene_dir = os.path.dirname(bpy.data.filepath)

    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            if obj.visible_get():
                obj.select_set(True)
            else:
                obj.select_set(False)

    for frame in range(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(frame)
        filepath = os.path.join(scene_dir, 'object/object_{:03}.obj'.format(frame))

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        bpy.ops.export_scene.obj(filepath=filepath, use_mesh_modifiers=True, use_selection=True,
                                 use_materials=False, use_triangles=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_start', type=int, default=1)
    parser.add_argument('--frame_end', type=int, default=5)
    parser.add_argument('--use_scene_frame_range', action='store_true')

    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    export_object(args.frame_start, args.frame_end, args.use_scene_frame_range)
