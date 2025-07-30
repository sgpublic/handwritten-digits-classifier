from vision_models.core.types.model_save_type import ModelSaveType
from vision_models.trainer.trainer_loader import load_trainer


def upload():
    trainer = load_trainer()
    trainer.load_weight(use_pretrained=False)
    trainer.upload_all()

if __name__ == '__main__':
    upload()