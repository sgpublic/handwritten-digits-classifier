from typing import Optional

from PIL import ImageOps

import gradio

from vision_models.core.model import Model
from vision_models.core.utils.logger import create_logger
from vision_models.gradio_app.gradio_app import GradioApp
from vision_models.models.mnist.mnist_model import MnistModel

logger = create_logger(__name__)

class MnistGradioApp(GradioApp):
    def _load_model(self) -> Model:
        return MnistModel()

    def _create_gradio(self):
        with gradio.Blocks() as demo:
            with gradio.Row():
                sketchpad = gradio.Sketchpad(
                    label="请在此处写一个 0 ~ 9 的数字",
                    image_mode="L",
                    type="pil",
                    show_download_button=False,
                    show_fullscreen_button=False,
                    layers=False,
                    fixed_canvas=True,
                    show_share_button=False,
                    height=420,
                    width=420,
                )
                result_box = gradio.Textbox(label="预测结果", interactive=False)
                sketchpad.change(self._predict, inputs=[sketchpad], outputs=result_box)
        return demo

    def _predict(self, sketchpad: dict) -> Optional[int]:
        if sketchpad["composite"] is None:
            return None
        predicted_number, _ = self._real_predict(ImageOps.invert(sketchpad["composite"]))
        return predicted_number
