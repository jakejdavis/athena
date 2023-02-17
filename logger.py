import colorlog
import logging

handler = colorlog.StreamHandler()

fmt = '[%(log_color)s%(levelname)s%(reset)s][%(name)s] %(message)s'
colorlog.basicConfig(level=logging.DEBUG, format=fmt)
logger = colorlog.getLogger(__name__)

def set_logger_level(verbose: bool) -> None:
    if verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debugging enabled")
    else:
        logger.setLevel(logging.INFO)