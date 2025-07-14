import gradio

from hwdc.core.logger import create_logger
from hwdc.hwdc_model import HwdcModel
from hwdc_gradio.config import HWDC_MODEL_USE_PRETRAINED, HWDC_GRADIO_PORT, HWDC_GRADIO_HOST

logger = create_logger(__name__)

class HwdcGradioApp:
    def __init__(self):
        self._gradio = self._create_gradio()
        self._model = HwdcModel()

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
                )
                result_box = gradio.Textbox(label="预测结果", interactive=False)
                sketchpad.change(self._predict, inputs=[sketchpad], outputs=result_box)
        return demo

    def launch(self):
        self._model.load(HWDC_MODEL_USE_PRETRAINED)
        self._gradio.launch(server_name=HWDC_GRADIO_HOST, server_port=HWDC_GRADIO_PORT)

    def _predict(self, sketchpad: dict) -> int | None:
        if sketchpad["composite"] is not None:
            processed_image = self._model.preprocess([sketchpad["composite"]])
            predicted_number, _ = self._model.predict(processed_image)[0]
            return predicted_number
        else:
            return None

