from distutils.util import strtobool
from os import environ

HWDC_MODEL_USE_PRETRAINED: bool = bool(strtobool(environ.get("HWDC_MODEL_USE_PRETRAINED", "true")))
