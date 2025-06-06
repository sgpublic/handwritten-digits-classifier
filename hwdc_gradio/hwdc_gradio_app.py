import gradio

from hwdc.hwdc_model import HwdcModel
from hwdc_gradio.config import HWDC_MODEL_USE_PRETRAINED


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
                sketchpad.change(self.predict, inputs=[sketchpad], outputs=result_box)
        return demo

    def launch(self):
        self._model.load(HWDC_MODEL_USE_PRETRAINED)
        self._gradio.launch()

    def predict(self, sketchpad: dict) -> int | None:
        if sketchpad["composite"] is not None:
            predicted_number, confidence = self._model.predict(sketchpad["composite"])
            return predicted_number
        else:
            return None

