from hwdc.hwdc_model import HwdcModel

if __name__ == '__main__':
    model = HwdcModel()
    model.load()
    model.export_as_onnx()
