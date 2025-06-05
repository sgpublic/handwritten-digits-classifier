import torch
from huggingface_hub import upload_file
from torch import nn, optim
from torch.nn.modules.loss import _WeightedLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from hwdc.core.config import HWDC_DATASET_EPOCHS, HWDC_DATASET_BATCH_SIZE
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
        logger.info("start training")
        train_loader = load_mnist_dataset(split="train", batch_size=batch_size)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self._resnet_model.parameters(), lr=1e-3)

        self._resnet_model.train()
        for index in range(epochs):
            avg_loss = self._real_train(train_loader=train_loader, criterion=criterion, optimizer=optimizer)
            print(f"epoch [{index + 1}/{epochs}], loss: {avg_loss:.4f}")
        self._resnet_model.eval()
        logger.info("training finished")

    def _real_train(self, train_loader: DataLoader, criterion: _WeightedLoss, optimizer: Optimizer) -> float:
        total_loss = 0
        for batch in train_loader:
            inputs, labels = batch["image"], batch["label"]
            inputs, labels = inputs.to(self._device), labels.to(self._device)

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
