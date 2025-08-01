import os
from distutils.util import strtobool

GRADIO_LISTEN_HOST: str = str(os.getenv("GRADIO_LISTEN_HOST", "0.0.0.0"))
GRADIO_LISTEN_PORT: int = int(os.getenv("GRADIO_LISTEN_PORT", "7860"))
GRADIO_MODEL_USE_PRETRAINED: bool = bool(strtobool(os.getenv("GRADIO_MODEL_USE_PRETRAINED", "true")))
