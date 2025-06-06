from distutils.util import strtobool
from os import environ

HWDC_DEBUG: bool = bool(strtobool(environ.get("HWDC_DEBUG", "false")))
HWDC_DEVICE: str = environ.get("HWDC_DEVICE", "cpu")
HWDC_DATASET_BATCH_SIZE: int = int(environ.get("HWDC_DATASET_BATCH_SIZE", "100"))
HWDC_DATASET_EPOCHS: int = int(environ.get("HWDC_EPOCHS", "10"))
HWDC_MODEL_SAVE_INTERVAL: int = int(environ.get("HWDC_MODEL_SAVE_INTERVAL", "1"))
