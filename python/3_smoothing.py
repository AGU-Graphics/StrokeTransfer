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


from util.base_cli import *
from util.base_option import *
from util.smoothing import out_smoothing
from util.transfer.vf_vis import out_vis_vf


class SmoothingOption(BaseOption):
    """ Parse the xml file used for vector field smoothing.

    Attributes:
        orientation_file_template: input vector field file template (.json) for smoothing.
        object_file_template: an instance of FileTemplateOption holding the 3D OBJ file template (.obj).
        
        lambda_spatial: regularization parameter for spatial coherence.
        lambda_temporal: regularization parameter for temporal coherence.
        deform: should be set to "True" if the target object is a deforming object. (False: the discrete hodge star, etc., will be computed using only the first frame; True: the discrete hodge star, etc., will be computed for every frame)

        diffuse_file_template: (used for visualization purpose only) an instance of FileTemplateOption holding the diffuse file template (.exr).
        camera_file_template: (used for visualization purpose only) an instance of FileTemplateOption holding the camera file template (.json).

        output_file_template: output file template (.json) for the results of vector field smoothing.

        frame_start: first frame number.
        frame_end: last frame number.
        frame_step: frame number step.

        verbose: if true, print intermediate info for the command.

        internal: directory path to save internal data for the command.
        internal_update: if True, internal data (if already exists due to e.g., a previous run) will be overwritten.
    """
    def __init__(self, root):
        ## inputs.
        self.orientation_file_template = root.find("orientation").attrib["filename_template"]
        self.object_file_template = FileTemplateOption("object", root)

        ## parameters.
        self.lambda_spatial = float(root.find("params").attrib["lambda_spatial"])
        self.lambda_temporal = float(root.find("params").attrib["lambda_temporal"])
        self.deform = bool(strtobool(root.find("params").attrib["deform"]))

        ## for visualization of vector field.
        self.diffuse_file_template = FileTemplateOption("diffuse", root)
        self.camera_file_template = FileTemplateOption("camera", root)

        ## outputs.
        self.output_file_template = root.find("output").attrib["orientation"]

        ## from BaseOption.
        self.load_frames_option(root)
        self.load_verbose_option(root)
        self.load_internal_option(root)

    def print_options(self):
        """ print option setting for the command."""

        print(f"[orientation file_name_template = {self.orientation_file_template}]")
        print(f"[object file_name_template = {self.object_file_template}]")
        print(f"[params lambda_spatial = {self.lambda_spatial}, lambda_temporal={self.lambda_temporal}, deform={self.deform}]")
        self.print_frames()
        self.print_verbose()
        print(f"[output file_name_template = {self.output_file_template}]")
        self.print_internal()


def cli_smoothing(option):
    """ command line interface for vector field smoothing.

    Args:
        option: an instance of SmoothingOption class used for smoothing command.
    """
    option.internal += "/transfer"
    out_smoothing(option)
    out_vis_vf(option, option.output_file_template, data_name="smooth_orientation", color=[0.122, 0.467, 0.706, 1.0], grid_step=1)


if __name__ == '__main__':
    print_cli_header("Vector Field Smoothing", datetime.date(2022, 7, 4), coded_by=CODER_TODO)

    run_cli(SmoothingOption, cli_smoothing)
