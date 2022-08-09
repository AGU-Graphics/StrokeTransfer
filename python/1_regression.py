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
from util.canonical_sections.out_canonical_sections import \
    output_canonical_sections
from util.features.out_features import output_features
from util.gbuffers.render_gbuffer import out_gbuffer
from util.logger import getLogger
from util.model.out_models import out_models
from util.model.out_weight import out_regression_weight_map

logger = getLogger(__name__)


class RegressionOption(BaseOption):
    """ Parse the xml file used for regression, to extract information regarding orientations, colors, lengths, and widths.

    Attributes:
        model_type_orientation: determine the order (recommended: 1) used for vector field regression.
        model_type_color: determine the model (recommended: "nearest-neighbor") used for color regression.
        model_type_length: determine the model (recommended: "nearest-neighbor") used for length regression.
        model_type_width: determine the model (recommended: "nearest-neighbor") used for width regression.

        exemplar_filename: the filename (in .png extension) of the raster image of the exemplar drawn by the artist.
        annotation_filename: annotation file name (.json).

        diffuse_file_name: an instance of FileNameOption holding the diffuse file name (.exr).
        specular_file_name: an instance of FileNameOption holding the specular file name (.exr).
        camera_file_name: an instance of FileNameOption holding the camera file name (.json).
        object_file_name: an instance of FileNameOption holding the 3D OBJ file name (.obj).

        model_output_orientation: output file name (in .pickle extension) for the result of vector field regression (the ".pickle" file dumps the content of the class instance).
        model_output_color: output file name (in .pickle extension) for the result of color regression (the ".pickle" file dumps the content of the class instance).
        model_output_length: output file name (in .pickle extension) for the result of length regression (the ".pickle" file dumps the content of the class instance).
        model_output_width: output file name (in .pickle extension) for the result of width regression (the ".pickle" file dumps the content of the class instance).

        frame_start: first frame number.
        frame_end: last frame number.
        frame_step: frame number step.

        verbose: if true, print intermediate info for the command.

        internal: directory path to save internal data for the command.
        internal_update: if True, internal data (if already exists due to e.g., a previous run) will be overwritten.
    """

    def __init__(self, root):
        """

        Args:
            root: the root node of the xml element tree (xml.etree.ElementTree).
        """
        ## model type.
        self.model_type_orientation = root.find("model_type").attrib["orientation"]
        self.model_type_color = root.find("model_type").attrib["color"]
        self.model_type_length = root.find("model_type").attrib["length"]
        self.model_type_width = root.find("model_type").attrib["width"]

        ## exemplar and annotation.
        self.exemplar_filename = root.find("exemplar").attrib["filename"]
        self.annotation_filename = root.find("annotation").attrib["filename"]

        ## reference information for regression.
        self.diffuse_file_name = FileNameOption("diffuse", root)
        self.specular_file_name = FileNameOption("specular", root)
        self.camera_file_name = FileNameOption("camera", root)
        self.object_file_name = FileNameOption("object", root)

        ## outputs.
        self.model_output_orientation = root.find("model_output").attrib["orientation"]
        self.model_output_color = root.find("model_output").attrib["color"]
        self.model_output_length = root.find("model_output").attrib["length"]
        self.model_output_width = root.find("model_output").attrib["width"]

        ## from BaseOption.
        self.dummy_frames_options()
        self.load_verbose_option(root)
        self.load_internal_option(root)

    def print_options(self):
        """ print option setting for the command."""

        print(
            f"[model_type orientation={self.model_type_orientation}, color={self.model_type_color}, length={self.model_type_length}, width={self.model_type_width}]")

        print(f"[exemplar filename={self.exemplar_filename}]")
        print(f"[annotation filename={self.annotation_filename}]")

        self.diffuse_file_name.print()
        self.specular_file_name.print()
        self.camera_file_name.print()
        self.object_file_name.print()

        self.print_verbose()
        print(
            f"[model_output orientation={self.model_output_orientation}, color={self.model_output_color}, length={self.model_output_length}, width={self.model_output_width}]")
        self.print_internal()


def cli_regression(option):
    """ command line interface for regression.

    Args:
        option: an instance of RegressionOption class used for regression command.
    """
    option.internal += "/regression"
    option.diffuse_file_template = option.diffuse_file_name
    option.specular_file_template = option.specular_file_name
    option.camera_file_template = option.camera_file_name
    option.object_file_template = option.object_file_name

    out_gbuffer(option)
    out_models(option)
    output_features(option)
    output_canonical_sections(option)
    out_regression_weight_map(option)


if __name__ == '__main__':
    print_cli_header("Regression", datetime.date(2022, 7, 8), coded_by=CODER_TODO)
    run_cli(RegressionOption, cli_regression)
