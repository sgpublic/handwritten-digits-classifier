import logging

from vision_models.core.config import CORE_DEBUG

if CORE_DEBUG:
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

