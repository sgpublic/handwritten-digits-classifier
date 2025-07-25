from hwdc_trainer.hwdc_model_trainer import HwdcModelTrainer

def upload():
    trainer = HwdcModelTrainer()
    trainer.load(use_pretrained=False)
    trainer.upload()
    trainer.upload_onnx()

if __name__ == '__main__':
    upload()