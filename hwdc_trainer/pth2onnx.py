from hwdc.hwdc_model import HwdcModel

if __name__ == '__main__':
    model = HwdcModel()
    if model.load(use_pretrained=False):
        model.export_as_onnx()
