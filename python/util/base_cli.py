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


import time
import datetime

from util.base_option import load_xml_option

CODER_TODO = "Hideki Todo <tody411@gmail.com>"
CODER_KOBAYASHI = "Kunihiko Kobayashi <kuni.koba.one-piece@outlook.com>"


def print_cli_header(cli_name, dates, version="1.0", coded_by=None):
    print("############################################################################")
    print(f"## Stroke Transfer: {cli_name.ljust(42)}            ##")
    print(f"## - Authors: Hideki Todo, Kunihiko Kobayashi, Jin Katsuragi, {''.ljust(12)}##")
    print(f"##            Haruna Shimotahira, Shizuo Kaji, Yonghao Yue    {''.ljust(12)}##")
    if coded_by is not None:
        print(f"## - Maintained by: {coded_by.ljust(52)}  ##")
    print(f"## - Dates: {dates.strftime('%Y/%m/%d').ljust(42)}                    ##")
    print(f"## - Version: {version.ljust(42)}                  ##")
    print("############################################################################")


def run_cli(option_cls, cli_fn):
    option = load_xml_option(option_cls)

    time_start = time.time()
    cli_fn(option)
    time_end = time.time()
    sec = time_end - time_start
    print(f"Total Time: {sec:.4f} sec")
