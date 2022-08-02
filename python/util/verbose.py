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
