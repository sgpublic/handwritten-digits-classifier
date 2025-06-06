import os

import torch
from huggingface_hub import upload_file
from torch import nn, optim, Tensor
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
        super().__init__(use_pretrained=False)

    def train(self,
              epochs: int = HWDC_DATASET_EPOCHS,
              batch_size: int = HWDC_DATASET_BATCH_SIZE):
        logger.info("prepare training")
        train_loader = load_mnist_dataset(split="train", batch_size=batch_size)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self._resnet_model.parameters(), lr=1e-3)

        logger.info("start training")
        self._resnet_model.train()
        save_interval = 0
        for index in range(epochs):
            logger.info(f"epoch [{index + 1}/{epochs}]...")
            avg_loss = self._real_train(train_loader=train_loader, criterion=criterion, optimizer=optimizer)
            logger.info(f"epoch [{index + 1}/{epochs}], loss: {avg_loss:.4f}")
            save_interval += 1
            if save_interval % HWDC_MODEL_SAVE_INTERVAL == 0:
                logger.info("save model weight...")
                save_interval = 0
                self._resnet_model.eval()
                self.save()
                self._resnet_model.train()
        self._resnet_model.eval()
        self.save()
        logger.info("training finished")

    def _real_train(self, train_loader: DataLoader, criterion: _WeightedLoss, optimizer: Optimizer) -> float:
        total_loss = 0
        for batch in train_loader:
            inputs: Tensor = batch["image"].to(self._device)
            labels: Tensor = batch["label"].to(self._device)

            optimizer.zero_grad()
            outputs = self._resnet_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def save(self) -> bool:
        if self._resnet_model.training:
            logger.error("model is still in training, please stop training and try again.")
            return False
        try:
            os.makedirs(name=os.path.dirname(self._model_local), exist_ok=True)
            torch.save(self._resnet_model.state_dict(), self._model_local)
            return True
        except Exception:
            logger.exception("weight save failed!")
            return False

    def upload(self):
        try:
            logger.exception("uploading model...")
            upload_file(
                path_or_fileobj=self._model_local,
                path_in_repo=self._repo_pretrained_model,
                repo_id=self._repo_id,
                repo_type="model",
            )
        except Exception:
            logger.exception("upload model failed")
