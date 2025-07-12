from hwdc_trainer.hwdc_model_trainer import HwdcModelTrainer


def start_train():
    trainer = HwdcModelTrainer()
    trainer.load(use_pretrained=False)
    trainer.train()

if __name__ == "__main__":
    start_train()
