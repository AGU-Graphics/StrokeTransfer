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


import tqdm


def verbose_print(verbose, message):
    """ Print message only if verbose option is True.

    Args:
        verbose: if True, the input message will be printed.
        message: input message to print.
    """
    if verbose:
        print(message)


def verbose_range(verbose, ranges, desc=None):
    """ Print progress bar messages for ranges only if verbose option is True.

    Args:
        verbose: if True, progress bar messages will be printed.
        ranges: input ranges to display progress bar messages.

    Returns:
        ranges: if verbose is True, progress bar message tqdm.tqdm(ranges) is returned.
    """
    if verbose:
        if desc is not None:
            ranges = tqdm.tqdm(ranges, desc=desc)
        else:
            ranges = tqdm.tqdm(ranges)
    return ranges
