from hwdc_gradio.hwdc_gradio_app import HwdcGradioApp


def start_webui():
    app = HwdcGradioApp()
    app.launch()


if __name__ == '__main__':
    start_webui()
    