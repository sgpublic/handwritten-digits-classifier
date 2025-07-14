from distutils.util import strtobool
from os import environ

HWDC_DEBUG: bool = bool(strtobool(environ.get("HWDC_DEBUG", "false")))
HWDC_DEVICE: str = environ.get("HWDC_DEVICE", "cpu")
HWDC_DATASET_BATCH_SIZE: int = int(environ.get("HWDC_DATASET_BATCH_SIZE", "100"))
HWDC_DATASET_TEST_DATASET_SIZE: int = int(environ.get("HWDC_DATASET_TEST_DATASET_SIZE", "1000"))
HWDC_DATASET_MAX_EPOCHS: int = int(environ.get("HWDC_DATASET_MAX_EPOCHS", "50"))
HWDC_MODEL_ACCURACY_THRESHOLD: float = float(environ.get("HWDC_MODEL_ACCURACY_THRESHOLD", "0.99"))
