from distutils.util import strtobool
from os import environ

from hwdc.hwdc_model_type import HwdcModelType

HWDC_DEBUG: bool = bool(strtobool(environ.get("HWDC_DEBUG", "false")))
HWDC_DEVICE: str = environ.get("HWDC_DEVICE", "cuda")
HWDC_MODEL_TYPE: HwdcModelType = HwdcModelType[environ.get("HWDC_MODEL_TYPE", HwdcModelType.ResNet.name)]
