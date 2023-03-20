import io
import logging

import colorlog
import tensorflow as tf

EXTERNAL_LOGGERS = ["tensorflow-metal", "matplotlib", "h5py", "h5py_.conv"]

handler = colorlog.StreamHandler()
fmt = "[%(log_color)s%(levelname)s%(reset)s][%(name)s][%(filename)s] %(message)s"
colorlog.basicConfig(level=logging.DEBUG, format=fmt)


def set_logger_level(verbose: bool) -> None:
    logger = logging.getLogger()
    logger.name = "athena"
    if verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debugging enabled")
    else:
        logger.setLevel(logging.INFO)

    disable_external_loggers()


def disable_external_loggers():
    tf.get_logger().setLevel(logging.WARNING)

    for external_logger in EXTERNAL_LOGGERS:
        external_logger = logging.getLogger(external_logger)
        external_logger.setLevel(logging.WARNING)
