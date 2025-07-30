from vision_models.trainer.trainer_loader import load_trainer


def start_train():
    trainer = load_trainer()
    trainer.load_weight(use_pretrained=False)
    trainer.train()

if __name__ == "__main__":
    start_train()
