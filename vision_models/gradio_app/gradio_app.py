from abc import ABC, abstractmethod
from logging import Logger

from PIL.Image import Image

from vision_models.core.log import Log
from vision_models.core.model import VisionClassifyModel
from vision_models.gradio_app.config import GRADIO_MODEL_USE_PRETRAINED, GRADIO_LISTEN_HOST, GRADIO_LISTEN_PORT


class GradioApp(Log, ABC):
    @property
    def __logger_name__(self) -> str:
        return __name__

    def __init__(self):
        super().__init__()
        self._gradio = self._create_gradio()
        self._model = self._load_model()

    @abstractmethod
    def _load_model(self) -> VisionClassifyModel:
        pass

    @abstractmethod
    def _create_gradio(self):
        pass

    def _real_predict(self, payload: Image) -> tuple[int, float]:
        processed_image = self._model.preprocess([payload])
        return self._model.predict(payload=processed_image)[0]

    def launch(self):
        self._model.load_weight(use_pretrained=GRADIO_MODEL_USE_PRETRAINED)
        self._gradio.launch(server_name=GRADIO_LISTEN_HOST, server_port=GRADIO_LISTEN_PORT)
