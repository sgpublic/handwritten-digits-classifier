import os

import torch
from huggingface_hub import upload_file
from torch import nn, optim
from torch.nn.modules.loss import _WeightedLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from hwdc.core.config import HWDC_DATASET_EPOCHS, HWDC_DATASET_BATCH_SIZE, HWDC_MODEL_SAVE_INTERVAL
from hwdc.core.logger import create_logger
from hwdc.hwdc_model import HwdcModel
from hwdc_trainer.dataset_loader import load_mnist_dataset

logger = create_logger(__name__)


class HwdcModelTrainer(HwdcModel):
    def __init__(self):
        super().__init__()

    def train(self,
              epochs: int = HWDC_DATASET_EPOCHS,
              batch_size: int = HWDC_DATASET_BATCH_SIZE):
        logger.info("prepare training")
        train_loader = load_mnist_dataset(split="train", batch_size=batch_size)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.resnet_model.parameters(), lr=1e-3)

        logger.info("start training")
        self.resnet_model.train()
        save_interval = 0
        best_loss = None
        for index in range(epochs):
            logger.info(f"epoch [{index + 1}/{epochs}]...")
            avg_loss = self._real_train(train_loader=train_loader, criterion=criterion, optimizer=optimizer)
            logger.info(f"epoch [{index + 1}/{epochs}], loss: {avg_loss:.6f}")
            save_interval += 1
            if best_loss is None:
                best_loss = avg_loss
            if best_loss < avg_loss:
                logger.info("loss increased, skip saving model weight")
                continue
            if save_interval >= HWDC_MODEL_SAVE_INTERVAL:
                logger.info("loss decreased, save model weight")
                save_interval = 0
                self.save()
        self.resnet_model.eval()
        self.save()
        logger.info("training finished")

    def _real_train(self, train_loader: DataLoader, criterion: _WeightedLoss, optimizer: Optimizer) -> float:
        total_loss = 0
        for batch in train_loader:
            inputs, labels = batch["image"], batch["label"]
            inputs, labels = self.move_to_device(inputs), self.move_to_device(labels)

            optimizer.zero_grad()
            outputs = self.resnet_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def save(self) -> bool:
        still_training = self.resnet_model.training
        if still_training:
            self.resnet_model.eval()
        try:
            os.makedirs(name=os.path.dirname(self.model_local), exist_ok=True)
            torch.save(self.resnet_model.state_dict(), self.model_local)
            return True
        except Exception:
            logger.exception("weight save failed!")
            return False
        finally:
            if still_training:
                self.resnet_model.train()

    def upload(self):
        try:
            logger.exception("uploading model...")
            upload_file(
                path_or_fileobj=self.model_local,
                path_in_repo=self.repo_pretrained_model,
                repo_id=self.repo_id,
                repo_type="model",
            )
        except Exception:
            logger.exception("upload model failed")
