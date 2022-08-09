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


import datetime
import igl
import json
import numpy as np
import os
import point_cloud_utils as pcu

from distutils.util import strtobool

import util.gl.shaders.front_light as front_light_shader

from util.base_cli import CODER_KOBAYASHI, print_cli_header, run_cli
from util.base_option import BaseOption
from util.blender_info import BlenderInfo
from util.gl.renderer import Renderer
from util.verbose import verbose_range


class AnchorPointsOption(BaseOption):
    """
    Parse the xml file used for anchor point generation

    Attributes:
        object_file_template: object file template (.json) for the (animated) 3D object sequence (please provide a sequence of obj files for the frames even if the object is static).
        camera_file_template: (for visualization purpose only) camera file name (.json) exported from blender (please use blender/export_camera.py for exporting this json file).

        num_levels: the number of levels in the hierarchy of anchor points.
        initial_radius_ratio: the initial radius used for Poisson disc sampling relative to the diagonal length of the bounding box.
        deform: if True, the object is a deforming object.
        coherence: (recommended: True) if False, the anchor points will be generated per frame randomly without coherence.

        output_file_template: output file template (.json) for the anchor points (if coherence is True, a single json file containing the anchor points common to all frames will be generated; if coherence is false, a json file will be generated for each frame)

        frame_start: first frame number.
        frame_end: last frame number.
        frame_step: frame number step.

        verbose: if true, print intermediate info for the command.

        internal: directory path to save internal data for the command.
    """

    def __init__(self, root):
        """
        Args:
            root: the root node of the xml tree
        """
        super().__init__(root)
        ## reference information for anchor points.
        self.object_file_template = root.find("object").attrib["filename_template"]
        self.camera_file_template = root.find("camera").attrib["filename_template"]

        ## parameters.
        self.num_levels = int(root.find("params").attrib["num_levels"])
        self.initial_radius_ratio = float(root.find("params").attrib["initial_radius_ratio"])
        self.deform = bool(strtobool(root.find("params").attrib["deform"]))
        self.coherence = bool(strtobool(root.find("params").attrib["coherence"]))

        ## outputs.
        self.output_file_template = root.find("output").attrib["filename"]

        ## from BaseOption.
        self.load_frames_option(root)
        self.load_verbose_option(root)
        self.load_internal_option(root)

    def print_options(self):
        """ print option setting for the command."""

        print(f"[object file_name_template = {self.object_file_template}]")
        print(f"[camera file_name_template = {self.camera_file_template}]")
        print(f"[params num_levels = {self.num_levels}, initial_radius_ratio = {self.initial_radius_ratio}, deform = {self.deform}, coherence = {self.coherence}]")
        self.print_frames()
        self.print_verbose()
        print(f"[output file_name = {self.output_file_template}]")
        self.print_internal()


def create_anchor_points(option):
    """ Create and save anchor point data with the given settings.

    Args:
        option: an instance of AnchorPointsOption class used for anchor points command.
    """
    
    # anchor_nums: counts of anchor points on each level.
    anchor_nums = []
    anchor_F_idx = []
    anchor_barycentric_coordinate = []
    # anchor_file_name: output file path (.json) for anchor point data.
    anchor_file_name = option.output_file_template % option.frame_start

    for level in verbose_range(option.verbose, range(option.num_levels)):
        for frame in verbose_range(option.verbose, option.frame_range()):
            anchor_image_file_name = f"{option.internal}/anchor_points_%d/anchor_points_%d_%03d.png" % (level + 1, level + 1, frame)
            
            # model, view projection matrix from blender information.
            camera_info = BlenderInfo(option.camera_file_template % frame)
            model_mat, view_mat, project_mat = camera_info.MVPMatrix()
            
            if not option.coherence:
                anchor_file_name = option.output_file_template % frame
                anchor_nums = []
                if not level == 0:
                    with open(anchor_file_name) as in_file:
                        anchor_data = json.load(in_file)
                    anchor_nums = (anchor_data["anchor_nums"])
                    anchor_F_idx = np.array(anchor_data["F_idx"])
                    anchor_barycentric_coordinate = np.array(anchor_data["barycentric_coordinate"])
            
            # mesh data (vertices, faces) from obj file.
            V, F = igl.read_triangle_mesh(option.object_file_template % frame)
            
            renderer = Renderer(im_width=1024, im_height=1024)
            renderer.add_mesh(V, F, np.array([1.0, 1.0, 1.0, 1.0]), shader=front_light_shader)
            
            # initial radius is determined by the diagonal length of bounding box.
            bbox = np.max(V, axis=0) - np.min(V, axis=0)
            bbox_diag = np.linalg.norm(bbox)
            radius = (option.initial_radius_ratio / 2 ** level) * bbox_diag
            
            # random_seed setting
            random_seed = option.frame_start if frame == option.frame_start or option.coherence else int(ticks(datetime.datetime.utcnow())) % 25204043
            
            # anchor point distribution (face indices, barycentric coordinates) is obtained by poisson disk sampling (point cloud library).
            sampling_F_idx, sampling_barycentric_coordinate = pcu.sample_mesh_poisson_disk(V, F, random_seed=random_seed, num_samples=-1, radius=radius, use_geodesic_distance=True)

            # deform=True and not frame_start: append anchor points which has the closest distance more than radius.
            if option.deform and frame != option.frame_start:
                anchor_points = pcu.interpolate_barycentric_coords(F, anchor_F_idx, anchor_barycentric_coordinate, V)
                new_sampling_points = pcu.interpolate_barycentric_coords(F, sampling_F_idx, sampling_barycentric_coordinate, V)

                nearest_dists_new_sampling_to_anchor, _ = pcu.k_nearest_neighbors(new_sampling_points, anchor_points, 1)
                use_sampling_idx = nearest_dists_new_sampling_to_anchor > radius

                anchor_F_idx = np.concatenate([anchor_F_idx, sampling_F_idx[use_sampling_idx]])
                anchor_barycentric_coordinate = np.concatenate([anchor_barycentric_coordinate, sampling_barycentric_coordinate[use_sampling_idx]])

            # level!=0: remove the closest point of new_anchor_points from decide_anchor_points, and then append the rest of them.
            elif not level == 0:
                anchor_points = pcu.interpolate_barycentric_coords(F, anchor_F_idx, anchor_barycentric_coordinate, V)
                new_sampling_points = pcu.interpolate_barycentric_coords(F, sampling_F_idx, sampling_barycentric_coordinate, V)

                _, nearest_idx_anchor_to_new_sampling = pcu.k_nearest_neighbors(anchor_points, new_sampling_points, 1)
                use_sampling_idx = np.setdiff1d(np.arange(new_sampling_points.shape[0]), nearest_idx_anchor_to_new_sampling)

                anchor_F_idx = np.concatenate([anchor_F_idx, sampling_F_idx[use_sampling_idx]])
                anchor_barycentric_coordinate = np.concatenate([anchor_barycentric_coordinate, sampling_barycentric_coordinate[use_sampling_idx]])
            else:
                anchor_F_idx = sampling_F_idx
                anchor_barycentric_coordinate = sampling_barycentric_coordinate

            # save anchor points data to the target file.
            if not option.deform or option.deform and frame == option.frame_end:
                anchor_nums.append(anchor_F_idx.shape[0])
                anchor_data = {"anchor_point_nums": anchor_nums, "F_idx": anchor_F_idx.tolist(), "barycentric_coordinate": anchor_barycentric_coordinate.tolist()}
                os.makedirs(os.path.dirname(anchor_file_name), exist_ok=True)
                with open(anchor_file_name, "w") as out_file:
                    json.dump(anchor_data, out_file, indent=4)
            
            # render the visualization of anchor point generation.
            anchor_points = pcu.interpolate_barycentric_coords(F, anchor_F_idx, anchor_barycentric_coordinate, V)
            renderer.points = []
            renderer.add_points(anchor_points, R=24.0 * 1.6 ** (-(level + 1)), C=np.array([235, 134, 126, 255]) / 255.0)
            renderer.setMVPMat(model_mat, view_mat, project_mat)
            os.makedirs(os.path.dirname(anchor_image_file_name), exist_ok=True)
            renderer.render(out_file=anchor_image_file_name, show_result=False)

            if not option.deform and option.coherence:
                break


def ticks(dt):
    """ Return the time state for random seed. """

    return (dt - datetime.datetime(1, 1, 1)).total_seconds() * 10000000


def cli_anchor_points(option):
    """ command line interface for create anchor generation.

    Args:
        option: an instance of AnchorPointsOption class used for anchor_points command.
    """
    option.internal += "/anchor_points"
    if not option.coherence and option.deform:
        print("We do support the setting of \"deform = True\" and \"coherent = False.\"")
    else:
        create_anchor_points(option)


if __name__ == '__main__':
    print_cli_header("Create Anchor Points", datetime.date(2022, 7, 11), coded_by=CODER_KOBAYASHI)

    run_cli(AnchorPointsOption, cli_anchor_points)
