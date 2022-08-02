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
