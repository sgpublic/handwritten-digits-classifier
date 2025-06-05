import logging

from hwdc.core.config import HWDC_DEBUG

if HWDC_DEBUG:
    logging_level = logging.DEBUG
else:
    logging_level = logging.INFO

logging.basicConfig(
    format="%(asctime)s %(levelname)s - [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S",
    level=logging_level,
)


def create_logger(name: str):
    return logging.getLogger(name)

