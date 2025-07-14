from distutils.util import strtobool
from os import environ

HWDC_GRADIO_HOST: str = str(environ.get("HWDC_GRADIO_HOST", "0.0.0.0"))
HWDC_GRADIO_PORT: int = int(environ.get("HWDC_GRADIO_PORT", "7860"))
HWDC_MODEL_USE_PRETRAINED: bool = bool(strtobool(environ.get("HWDC_MODEL_USE_PRETRAINED", "true")))
