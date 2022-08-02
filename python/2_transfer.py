from util.base_cli import *
from util.base_option import *
from util.canonical_sections.out_canonical_sections import \
    output_canonical_sections
from util.features.out_features import output_features
from util.gbuffers.render_gbuffer import out_gbuffer
from util.model.out_weight import out_transfer_weight_map
from util.transfer.out_transfer import out_transfer, vis_lw_transfers


class TransferOption(BaseOption):
    """ Parse the xml file used for transfer, to extract information regarding orientations, colors, lengths, and widths.

    Attributes:
        model_orientation: file name (.pickle) storing the orientation model.
        model_color: file name (.pickle) storing the color model.
        model_length: file name (.pickle) storing the length model.
        model_width: file name (.pickle) storing the width model.

        diffuse_file_template: an instance of FileTemplateOption holding the diffuse file template (.exr).
        specular_file_template: an instance of FileTemplateOption holding the specular file template (.exr).
        camera_file_template: an instance of FileTemplateOption holding the camera file template (.json).
        object_file_template: an instance of FileTemplateOption holding the 3D OBJ file template (.obj).

        output_orientation_template: output file template (.json) for the sequence of transferred orientation fields.
        output_color_template: output file template (.png) for sequence of transferred color fields.
        output_length_template: output file template (.exr) for sequence of transferred length fields.
        output_width_template: output file template (.exr) for sequence of transferred width fields.

        frame_start: first frame number.
        frame_end: last frame number.
        frame_step: frame number step.

        verbose: if true, print intermediate info for the command.

        internal: directory path to save internal data for the command.
        internal_update: if True, internal data (if already exists due to e.g., a previous run) will be overwritten.
    """

    def __init__(self, root):
        ## models.
        self.model_orientation = root.find("model").attrib["orientation"]
        self.model_color = root.find("model").attrib["color"]
        self.model_length = root.find("model").attrib["length"]
        self.model_width = root.find("model").attrib["width"]

        ## reference information for transfer.
        self.diffuse_file_template = FileTemplateOption("diffuse", root)
        self.specular_file_template = FileTemplateOption("specular", root)
        self.camera_file_template = FileTemplateOption("camera", root)
        self.object_file_template = FileTemplateOption("object", root)

        ## outputs.
        self.output_orientation_template = root.find("output").attrib["orientation"]
        self.output_color_template = root.find("output").attrib["color"]
        self.output_length_template = root.find("output").attrib["length"]
        self.output_width_template = root.find("output").attrib["width"]

        ## from BaseOption.
        self.load_frames_option(root)
        self.load_verbose_option(root)
        self.load_internal_option(root)

    def print_options(self):
        """ print option setting for the command."""

        print(
            f"[model orientation={self.model_orientation}, color={self.model_color}, length={self.model_length}, width={self.model_length}]")

        self.diffuse_file_template.print()
        self.specular_file_template.print()
        self.camera_file_template.print()
        self.object_file_template.print()

        self.print_frames()
        self.print_verbose()

        print(
            f"[output orientation={self.output_orientation_template}, color={self.output_color_template}, length={self.output_length_template}, width={self.output_width_template}]")

        self.print_internal()


def cli_transfer(option):
    """ command line interface for transfer.

    Args:
        option: an instance of TransferOption class used for regression command.
    """
    option.internal += "/transfer"
    out_gbuffer(option)
    out_transfer(option)
    output_features(option)
    output_canonical_sections(option)
    vis_lw_transfers(option)
    out_transfer_weight_map(option)


if __name__ == '__main__':
    print_cli_header("Transfer Stroke Attributes", datetime.date(2022, 7, 8), coded_by=CODER_TODO)
    run_cli(TransferOption, cli_transfer)
