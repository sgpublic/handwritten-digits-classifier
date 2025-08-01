from typing import Optional

import gradio

from vision_models.core.model import VisionClassifyModel
from vision_models.core.utils.logger import create_logger
from vision_models.gradio_app.gradio_app import GradioApp
from vision_models.models.cifar10.cifar10_model import Cifar10Model

logger = create_logger(__name__)

class Cifar10GradioApp(GradioApp):
    def _load_model(self) -> VisionClassifyModel:
        return Cifar10Model()

    def _create_gradio(self):
        with gradio.Blocks() as demo:
            pass
        return demo

    def _predict(self, sketchpad: dict) -> Optional[any]:
        pass
