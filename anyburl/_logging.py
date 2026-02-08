"""Centralized logging configuration for AnyBURL."""

import logging

PACKAGE_LOGGER_NAME: str = "anyburl"


def get_logger(name: str) -> logging.Logger:
    """Get a logger namespaced under ``anyburl``.

    All loggers are children of the ``anyburl`` logger so that users
    can configure logging for the entire package at once via
    ``logging.getLogger("anyburl")``.

    Parameters
    ----------
    name : str
        Module name, typically passed as ``__name__``.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    short_name = name.rsplit(".", maxsplit=1)[-1]
    return logging.getLogger(f"{PACKAGE_LOGGER_NAME}.{short_name}")
