# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import atexit as _atexit
import sys
from typing import Literal

import jax
from loguru._logger import Core as _Core  # pyright: ignore
from loguru._logger import Logger as _Logger  # pyright: ignore

LOG: bool = False
"""Whether to log or not. This is set to True when the logger is enabled."""


class Logger(_Logger):
    def trace(__self, __message, *args, **kwargs):  # noqa: N805
        r"""Log ``message.format(*args, **kwargs)`` with severity ``'TRACE'``."""
        if LOG:
            jax.debug.callback(
                __self._log,
                level="TRACE",
                from_decorator=False,
                options=__self._options,
                message=__message,
                args=args,
                kwargs=kwargs,
            )

    def debug(__self, __message, *args, **kwargs):  # noqa: N805
        r"""Log ``message.format(*args, **kwargs)`` with severity ``'DEBUG'``."""
        if LOG:
            jax.debug.callback(
                __self._log,
                level="DEBUG",
                from_decorator=False,
                options=__self._options,
                message=__message,
                args=args,
                kwargs=kwargs,
            )

    def info(__self, __message, *args, **kwargs):  # noqa: N805
        r"""Log ``message.format(*args, **kwargs)`` with severity ``'INFO'``."""
        if LOG:
            jax.debug.callback(
                __self._log,
                level="INFO",
                from_decorator=False,
                options=__self._options,
                message=__message,
                args=args,
                kwargs=kwargs,
            )

    def success(__self, __message, *args, **kwargs):  # noqa: N805
        r"""Log ``message.format(*args, **kwargs)`` with severity ``'SUCCESS'``."""
        if LOG:
            jax.debug.callback(
                __self._log,
                level="SUCCESS",
                from_decorator=False,
                options=__self._options,
                message=__message,
                args=args,
                kwargs=kwargs,
            )

    def warning(__self, __message, *args, **kwargs):  # noqa: N805
        r"""Log ``message.format(*args, **kwargs)`` with severity ``'WARNING'``."""
        if LOG:
            jax.debug.callback(
                __self._log,
                level="WARNING",
                from_decorator=False,
                options=__self._options,
                message=__message,
                args=args,
                kwargs=kwargs,
            )

    def error(__self, __message, *args, **kwargs):  # noqa: N805
        r"""Log ``message.format(*args, **kwargs)`` with severity ``'ERROR'``."""
        if LOG:
            jax.debug.callback(
                __self._log,
                level="ERROR",
                from_decorator=False,
                options=__self._options,
                message=__message,
                args=args,
                kwargs=kwargs,
            )

    def critical(__self, __message, *args, **kwargs):  # noqa: N805
        r"""Log ``message.format(*args, **kwargs)`` with severity ``'CRITICAL'``."""
        if LOG:
            jax.debug.callback(
                __self._log,
                level="CRITICAL",
                from_decorator=False,
                options=__self._options,
                message=__message,
                args=args,
                kwargs=kwargs,
            )

    def exception(__self, __message, *args, **kwargs):  # noqa: N805
        r"""Log an ``'ERROR'```` message while also capturing the currently handled
        exception.
        """
        if LOG:
            options = (True,) + __self._options[1:]
            jax.debug.callback(
                __self._log,
                level="ERROR",
                from_decorator=False,
                options=options,
                message=__message,
                args=args,
                kwargs=kwargs,
            )

    def log(__self, __level, __message, *args, **kwargs):  # noqa: N805
        r"""Log ``message.format(*args, **kwargs)`` with severity ``level``."""
        if LOG:
            jax.debug.callback(
                __self._log,
                level=__level,
                from_decorator=False,
                options=__self._options,
                message=__message,
                args=args,
                kwargs=kwargs,
            )


logger = Logger(
    core=_Core(),
    exception=None,
    depth=0,
    record=False,
    lazy=False,
    colors=False,
    raw=False,
    capture=True,
    patchers=[],
    extra={},
)
"""Logger instance for logging messages.

This logger is a singleton and can be used to log messages at different levels.
The logger is configured to log messages to the standard output stream with a specific
format. The log level can be set to different levels such as TRACE, DEBUG, INFO,
SUCCESS, WARNING, ERROR, and CRITICAL.

To enable logging, use the :func:`enable_logging` function.

It is a modified form of `loguru` logger to work with JAX. For more details on the
logger, refer to the `loguru documentation <https://loguru.readthedocs.io>`_.

Example
-------
```python
from flowMC.logger import logger, enable_logging
enable_logging(log_level="INFO")
logger.info("This is an info message.")
logger.error("This is an error message.")
```
"""


_atexit.register(logger.remove)


def enable_logging(
    log_level: Literal[
        "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
    ] = "TRACE",
) -> None:
    """Enable logging with the specified log level.

    Parameters
    ----------
    log_level : Literal[ &quot;TRACE&quot;, &quot;DEBUG&quot;, &quot;INFO&quot;, &quot;SUCCESS&quot;, &quot;WARNING&quot;, &quot;ERROR&quot;, &quot;CRITICAL&quot; ], optional
        The log level to use, by default &quot;TRACE&quot;
    """
    logger.remove()
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<level>{message}</level>",
    )
    global LOG
    LOG = True
