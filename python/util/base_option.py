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
import xml.etree.ElementTree as ET
from distutils.util import strtobool


class BaseOption:
    """ Parse the xml file used in common among different tools.
        The xml parser for each tool should inherit this class.

    Attributes:
        frame_start: first frame number.
        frame_end: last frame number.
        frame_step: frame number step.

        verbose: if true, print intermediate info.

        internal: directory path to save internal data.
        internal_overwrite: if True, internal data (if already exists due to e.g., a previous run) will be overwritten.
    """

    def __init__(self, root):
        """

        Args:
            root: the root node of the xml element tree (xml.etree.ElementTree).
        """
        pass

    def load_internal_option(self, root):
        self.internal = root.find("internal").attrib["dirname"]
        self.internal_overwrite = False

        if "overwrite" in root.find("internal").attrib:
            self.internal_overwrite = bool(strtobool(root.find("internal").attrib["overwrite"]))

    def load_frames_option(self, root):
        self.frame_start = int(root.find("frames").attrib["start"])
        self.frame_end = int(root.find("frames").attrib["end"])

        if "step" in root.find("frames").attrib:
            self.frame_step = int(root.find("frames").attrib["step"])
        else:
            self.frame_step = 1

    def dummy_frames_options(self, frame=0):
        self.frame_start = frame
        self.frame_end = frame
        self.frame_step = 1

    def frame_range(self):
        return range(self.frame_start, self.frame_end + 1, self.frame_step)

    def load_verbose_option(self, root):
        self.verbose = bool(strtobool(root.find("verbose").attrib["on"]))

    def print(self):
        self.print_header()
        self.print_options()
        print("")

    def print_header(self):
        print("xml:")

    def print_internal(self):
        info = f"[internal dirname={self.internal}"

        if self.internal_overwrite is not None:
            info += f", overwrite={self.internal_overwrite}"
        info += "]"
        print(info)

    def print_frames(self):
        print(f"[frames start = {self.frame_start}, end={self.frame_end}]")

    def print_verbose(self):
        print(f"[verbose on = {self.verbose}]")

    def print_options(self):
        pass

    def load_option_file_name(self, option_file_name):
        self.option_file_name = option_file_name


class FileTemplateOption:
    """ Utility class for generating the filename (e.g., "dst/diffuse/diffuse_001.exr") given a file name template (e.g., "dst/diffuse/diffuse_%03d.exr") and the frame number (e.g., 1).

    Attributes:
        name: xml element name.
        file_template: e.g., "dst/diffuse/diffuse_%03d.exr".
        file_extension: e.g., ".exr".
    """

    def __init__(self, name, root=None, file_template=None):
        self.name = name

        if file_template is None:
            self.file_template = root.find(name).attrib["filename_template"]
        else:
            self.file_template = file_template
        self.file_extension = os.path.splitext(self.file_template)[-1][1:]

    def file(self, frame):
        return self.file_template % frame

    def print(self):
        print(f"[{self.name} filename_template={self.file_template}]")


class FileNameOption:
    """ Utility class for holding a file name (e.g., exemplar file name, diffuse file name for regression, etc).

    Attributes:
        name: xml element name.
        file_name: e.g., "dst/diffuse/diffuse_%03d.exr".
        file_extension: e.g., ".exr".
    """

    def __init__(self, name, root):
        self.name = name
        self.file_name = root.find(name).attrib["filename"]
        self.file_extension = os.path.splitext(self.file_name)[-1][1:]

    def file(self, frame=None):
        return self.file_name

    def print(self):
        print(f"[{self.name} filename={self.file_name}]")


def load_xml_option(option_cls, xml_file_name=None, print_on=True):
    """ 
    
    Args:
        option_cls: a type of a class inheriting BaseOption
        xml_file_name: xml file name
        print_on: if True, print the formatted strings for xml attribute name-value pairs

    Returns:
        option: an instance of a class specified with the option_cls.
    """
    if xml_file_name is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('setting', help='Input xml setting file.')
        args = parser.parse_args()
        xml_file_name = args.setting

    tree = ET.parse(xml_file_name)
    root = tree.getroot()

    option = option_cls(root)
    option.load_option_file_name(xml_file_name)

    if print_on:
        print(f"Setting File: {xml_file_name}")
        option.print()
    return option
