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


import json
import os

import igl
import numpy as np

from util.blender_info import BlenderInfo
from util.logger import getLogger

logger = getLogger(__name__)


def load_scene_data(option, frame):
    json_file = option.camera_file_template.file(frame)
    blender_info = BlenderInfo(json_file)
    model_mat, view_mat, project_mat = blender_info.MVPMatrix()

    obj_file = option.object_file_template.file(frame)
    V, F = igl.read_triangle_mesh(obj_file)

    return V, F, model_mat, view_mat, project_mat


def load_orientation(file_template, frame=1):
    in_file = file_template % frame

    logger.debug(f"load_orientation: {in_file}")

    if not os.path.exists(in_file):
        return None

    with open(in_file) as fp:
        json_data = json.load(fp)

    c_e = np.array(json_data["orientation"])
    return c_e


def save_orientation(file_template, c_e, frame=1):
    out_c_e_file = file_template % frame

    out_dir = os.path.dirname(out_c_e_file)

    os.makedirs(out_dir, exist_ok=True)

    with open(out_c_e_file, 'w') as fp:
        json.dump({"orientation": c_e.tolist()}, fp, indent=4)


def internal_file(option, data_name, frame, format="png", dir_name=None):
    if dir_name is None:
        out_dir = f"{option.internal}/{data_name}"
    else:
        out_dir = f"{option.internal}/{dir_name}/{data_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{data_name}_{frame:03d}.{format}"
    return out_path
