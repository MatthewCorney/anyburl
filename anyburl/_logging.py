"""Logging and progress reporting for AnyBURL."""

import logging
from collections.abc import Collection, Iterator

from tqdm import tqdm

PACKAGE_LOGGER_NAME: str = "anyburl"


def get_logger(name: str) -> logging.Logger:
    """Get the logger for a module of the ``anyburl`` package.

    Loggers follow the module hierarchy, so ``logging.getLogger("anyburl")``
    configures the whole package and ``"anyburl.metrics"`` one subpackage.

    Parameters
    ----------
    name : str
        Module name, typically passed as ``__name__``.

    Returns
    -------
    logging.Logger
        The module's logger.
    """
    return logging.getLogger(name)


def progress[T](items: Collection[T], *, desc: str) -> Iterator[T]:
    """Iterate ``items`` with a progress bar on an interactive terminal.

    The bar is hidden when output is not a terminal, such as a log file,
    and when ``items`` is empty.

    Parameters
    ----------
    items : Collection[T]
        The items to iterate.
    desc : str
        Label shown beside the bar.

    Returns
    -------
    Iterator[T]
        The items, in order.
    """
    return iter(tqdm(items, desc=desc, disable=None if items else True))
