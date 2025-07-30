from distutils.util import strtobool
from os import environ

GRADIO_LISTEN_HOST: str = str(environ.get("GRADIO_LISTEN_HOST", "0.0.0.0"))
GRADIO_LISTEN_PORT: int = int(environ.get("GRADIO_LISTEN_PORT", "7860"))
GRADIO_MODEL_USE_PRETRAINED: bool = bool(strtobool(environ.get("GRADIO_MODEL_USE_PRETRAINED", "true")))
