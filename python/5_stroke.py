import datetime
import imageio
import numpy as np
import os
import subprocess

from distutils.util import strtobool

from util.base_cli import CODER_KOBAYASHI, print_cli_header, run_cli
from util.base_option import BaseOption
from util.verbose import verbose_print, verbose_range


class StrokeOption(BaseOption):
    """ Parse the xml file used for stroke generation

    Attributes:
        object_file_template: object file template (.json) for the (animated) 3D object sequence (please provide a sequence of obj files for the frames even if the object is static).
        camera_file_template: camera file name (.json) exported from blender (please use blender/export_camera.py for exporting this json file).
        anchor_file_template: anchor file template (.json) for the anchor points (although we only have a single json file when the coherence is turned on True, please always specify the file name as a template e.g., "...%03d.json")
        orientation_file_template: orientation file template (.json) from vector field smoothing.
        color_file_template: color file template (.png) from transfer.
        length_file_template: length file template (.exr) from transfer.
        width_file_template: width file template (.exr) from transfer.
        undercoat_file_template: raster image file template (.png) for undercoat (default: same as color_file_template, but can be a different image sequence, e.g., for multi layer synthesis).
        texture_file: file name of the brush texture image (.png).

        max_random_angular_offset: maximum random angular offset.
        resume: boolean option (True: do not overwrite existing output images and continue the rendering for the rest of the frames; False: generate final stroke renderings from the first frame and overwrite existing images).
        num_levels: the number of levels in the hierarchy of anchor points.

        length_multiplier: a (global) factor to amplify the length (usually set to 1.0 to disable amplification).
        length_offset: an (global) offset to the length (usually set to 0.0 to disable offset). (If "length_multiplier=0.0", then the length of strokes will be a constant value equal to this offset.)
        width_multiplier: a (global) factor to amplify the width (usually set to 1.0 to disable amplification).
        width_offset: an (global) offset to the width (usually set to 0.0 to disable offset). (If "width_multiplier=0.0", then the width of strokes will be a constant value equal to this offset.)
        coherence: (NEED TO BE EQUAL TO THE COHERENCE OPTION USED FOR ANCHOR POINT GENERATION) (recommended: True) set True to enable stroke sorting.

        drawing_incremental: boolean option (True: this "incremental mode" is used for the Max Planck example in our video, where the output images start from no stroke and the later frames with a fixed number of strokes added incrementally (sort of time-lapse)).
        drawing_num_incremental_strokes_per_frame: the number of strokes added incrementally in the "incremental mode".
        drawing_start_frame: the first frame we start the "incremental mode".

        output_stroke_template: output file template (.png) for the rendered images (without the undercoat layer).
        output_final_template: output file template (.png) for the rendered images (with the undercoat layer).

        frame_start: first frame number.
        frame_end: last frame number.
        frame_step: frame number step.

        verbose: if true, print intermediate info for the command.

        internal: directory path to save internal data for the command.
        internal_overwrite: if True, internal data (if already exists due to e.g., a previous run) will be overwritten.
    """

    def __init__(self, root):
        """
        Args:
            root: the root node of the xml tree
        """
        super().__init__(root)
        ## reference information for stroke.
        self.object_file_template = root.find("object").attrib["filename_template"]
        self.camera_file_template = root.find("camera").attrib["filename_template"]

        ## input images.
        self.anchor_file_template = root.find("anchor").attrib["filename_template"]
        self.orientation_file_template = root.find("orientation").attrib["filename_template"]
        self.color_file_template = root.find("color").attrib["filename_template"]
        self.length_file_template = root.find("length").attrib["filename_template"]
        self.width_file_template = root.find("width").attrib["filename_template"]
        self.undercoat_file_template = root.find("undercoat").attrib["filename_template"]
        self.texture_file = root.find("texture").attrib["filename"]

        ## parameters.
        self.max_random_angular_offset = float(root.find("params").attrib["max_random_angular_offset"])
        self.resume = bool(strtobool(root.find("params").attrib["resume"]))
        self.num_levels = int(root.find("params").attrib["num_levels"])

        ## optional parameters.
        self.length_multiplier = float(root.find("optional_params").attrib["length_multiplier"])
        self.length_offset = float(root.find("optional_params").attrib["length_offset"])
        self.width_multiplier = float(root.find("optional_params").attrib["width_multiplier"])
        self.width_offset = float(root.find("optional_params").attrib["width_offset"])
        self.coherence = bool(strtobool(root.find("optional_params").attrib["coherence"]))

        ## "incremental mode" parameters.
        self.drawing_incremental = bool(strtobool(root.find("params_drawing").attrib["incremental"]))
        self.drawing_num_incremental_strokes_per_frame = int(root.find("params_drawing").attrib["num_incremental_strokes_per_frame"])
        self.drawing_start_frame = int(root.find("params_drawing").attrib["start_frame"])

        ## outputs.
        self.output_stroke_template = root.find("output").attrib["stroke"]
        self.output_final_template = root.find("output").attrib["final"]

        ## from BaseOption.
        self.load_frames_option(root)
        self.load_verbose_option(root)
        self.load_internal_option(root)

    def print_options(self):
        """ print option setting for the command."""

        print(f"[object file_name_template = {self.object_file_template}]")
        print(f"[camera file_name_template = {self.camera_file_template}]")
        print(f"[anchor file_name = {self.anchor_file_template}]")
        print(f"[orientation file_name_template = {self.orientation_file_template}]")
        print(f"[color file_name_template = {self.color_file_template}]")
        print(f"[length file_name_template = {self.length_file_template}]")
        print(f"[width file_name_template = {self.width_file_template}]")
        print(f"[undercoat file_name_template = {self.undercoat_file_template}]")
        print(f"[texture file_name = {self.texture_file}]")
        print(f"[params max_random_angular_offset = {self.max_random_angular_offset}, resume = {self.resume}, num_levels = {self.num_levels}]")
        print(f"[optional_params length_multiplier = {self.length_multiplier}, length_offset = {self.length_offset}, width_multiplier = {self.width_multiplier}, width_offset = {self.width_offset}, coherence = {self.coherence}]")
        print(f"[params_drawing incremental = {self.drawing_incremental}, num_incremental_strokes_per_frame = {self.drawing_num_incremental_strokes_per_frame}, start_frame = {self.drawing_start_frame}]")
        self.print_frames()
        self.print_verbose()
        print(f"[output stroke = {self.output_stroke_template}, final = {self.output_final_template}]")
        self.print_internal()


def cli_stroke(option):
    """ command line interface for stroke generation.

    Args:
        option: an instance of StrokeOption class used for stroke command.
    """
    
    # cmd_temp: command line string template.
    cmd_temp = "python stroke_per_single_frame.py"

    for frame in verbose_range(option.verbose, option.frame_range()):
        # cmd: command line string.
        cmd = cmd_temp + f" {frame} {option.option_file_name}"
        verbose_print(option.verbose, cmd)
        if option.resume and os.path.exists(option.output_stroke_template % frame):
            continue

        subprocess.call(cmd.split(" "))
        
        # undercoat_image
        undercoat_image = imageio.imread(option.undercoat_file_template % frame).astype(np.float64) / 255.
        # stroke_image rendered images (without the undercoat layer)
        stroke_image = imageio.imread(option.output_stroke_template % frame).astype(np.float64) / 255.
        
        # final_image_rgb: RGB channel set of rendered images (with the undercoat layer)
        final_image_rgb = np.einsum("ijk,ij->ijk", undercoat_image[:, :, :3], 1 - stroke_image[:, :, 3]) + np.einsum("ijk,ij->ijk", stroke_image[:, :, :3], stroke_image[:, :, 3])
        # final_image_alpha: Alpha channel of rendered images (with the undercoat layer)
        final_image_alpha = np.einsum("ij,ij->ij", undercoat_image[:, :, 3], 1 - stroke_image[:, :, 3]) + stroke_image[:, :, 3]
        # final_image: full RGBA channel set of rendered images (with the undercoat layer)
        final_image = np.dstack((final_image_rgb, final_image_alpha))

        os.makedirs(os.path.dirname(option.output_final_template % frame), exist_ok=True)
        if option.output_final_template % frame is not None:
            imageio.imwrite(option.output_final_template % frame, np.uint8(255 * final_image))


if __name__ == '__main__':
    print_cli_header("Rendering Stroke", datetime.date(2022, 7, 11), coded_by=CODER_KOBAYASHI)

    run_cli(StrokeOption, cli_stroke)
