from util.base_cli import *
from util.base_option import *
from util.logger import getLogger
from util.tool import annotation_tool

logger = getLogger(__name__)


class AnnotationOption(BaseOption):
    """ Parse the xml file used for annotation.

    Attributes:
        exemplar_filename: the filename (in .png extension) of the raster image of the exemplar drawn by the artist.
    """

    def __init__(self, root):
        self.exemplar_filename = root.find("exemplar").attrib["filename"]

        self.load_verbose_option(root)
        self.load_internal_option(root)

    def print_options(self):
        """ print option setting for the command."""
        print(f"[exemplar filename={self.exemplar_filename}]")

        self.print_verbose()
        self.print_internal()


def cli_annotation(option):
    """ command line interface for annotation.

    Args:
        option: AnnotationOption used for annotation command.

    "0_annotation.py" is a command line tool that takes an xml file and launches the gui annotation tool.
    """
    exemplar_filename = os.path.abspath(option.exemplar_filename)

    module_dir = os.path.abspath(os.path.dirname(__file__))
    tool_dir = os.path.join(module_dir, "util/tool")

    os.chdir(tool_dir)
    annotation_tool.main(exemplar_filename)


if __name__ == '__main__':
    print_cli_header("Annotation", datetime.date(2022, 7, 18), coded_by=CODER_TODO)
    run_cli(AnnotationOption, cli_annotation)
