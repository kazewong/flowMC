# Copyright (c) 2022 Kaze Wong & contributor


import os

import jax.debug

DEBUG_VERBOSE = False


def enable_debug_verbose(verbose=True) -> None:
    r"""Enables verbose in debugging mode."""
    if not verbose:
        verbose = os.getenv("FLOWMC_DEBUG_VERBOSE", 0)
    global DEBUG_VERBOSE
    DEBUG_VERBOSE = bool(verbose)


def get_mode() -> bool:
    r"""Returns the current debugging verbose mode."""
    return DEBUG_VERBOSE


def flush(fmt_str: str, **kwargs) -> None:
    """Flushes the debug message to the console.

    .. code:: python
        >>> flush("Hello, {x}!", x=10)
        flowMC.Hello, 10!

    :param fmt_str: The format string for the debug message.
    """
    if DEBUG_VERBOSE:
        jax.debug.print("\033[1;30;42mflowMC.\033[0m" + fmt_str, **kwargs)
