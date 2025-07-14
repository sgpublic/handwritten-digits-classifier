import os

import numpy
import torch
from huggingface_hub import upload_file
from torch import nn, optim
from torch.nn.modules.loss import _WeightedLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from hwdc.core.config import HWDC_DATASET_MAX_EPOCHS, HWDC_DATASET_BATCH_SIZE, \
    HWDC_DATASET_TEST_DATASET_SIZE, HWDC_MODEL_ACCURACY_THRESHOLD
from hwdc.core.logger import create_logger
from hwdc.core.strings import arr2str, float2str
from hwdc.hwdc_model import HwdcModel
from hwdc_trainer.dataset_loader import mnist_dataset_loader

logger = create_logger(__name__)


class HwdcModelTrainer(HwdcModel):
    def __init__(self):
        super().__init__()

    def train(self,
              epochs: int = HWDC_DATASET_MAX_EPOCHS,
              batch_size: int = HWDC_DATASET_BATCH_SIZE,
              test_dataset_size: int = HWDC_DATASET_TEST_DATASET_SIZE,
              accuracy_threshold: float = HWDC_MODEL_ACCURACY_THRESHOLD):
        logger.info("prepare training")
        train_loader = mnist_dataset_loader(split="train", batch_size=batch_size)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.resnet_model.parameters(), lr=1e-3)

        logger.info("start training")
        self.resnet_model.train()
        best_loss = None
        for index in range(epochs):
            logger.info(f"epoch [{index + 1}/{epochs}]...")
            avg_loss = self._real_train(train_loader=train_loader, criterion=criterion, optimizer=optimizer)
            logger.info("testing...")
            avg_accuracy, current_accuracy = self._test(test_size=test_dataset_size)
            logger.info(f"epoch [{index + 1}/{epochs}], loss: {avg_loss:.6f}, "
                        f"accuracy({float2str(avg_accuracy)} in average): {arr2str(current_accuracy)}")

            if best_loss is None:
                best_loss = avg_loss
            if best_loss < avg_loss:
                logger.info("loss increased, skip saving model weight")
                continue
            logger.info("loss decreased, save model weight")
            self.save()

            # 要求每个数字的准确率都达到预期
            if all(x >= accuracy_threshold for x in current_accuracy):
            # 要求平均准确率达到预期
            # if avg_accuracy >= accuracy_threshold:
                logger.info("When the accuracy requirement is met, stop training.")
                break
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

    def _test(self, test_size: int = 1000) -> tuple[float, list[float]]:
        with torch.no_grad():
            total_count = numpy.zeros(10, dtype=numpy.int32)
            total_correct = numpy.zeros(10, dtype=numpy.int32)
            test_loader = mnist_dataset_loader(split="test", batch_size=test_size, dataset_size=test_size)
            # 实际上这个循环只跑一次
            for batch in test_loader:
                inputs, labels = batch["image"], batch["label"].tolist()
                outputs = self.predict(inputs)
                for (a, _), b in zip(outputs, labels):
                    total_count[b] += 1
                    if a == b:
                        total_correct[a] += 1
            return float(total_correct.sum() / total_count.sum()), [float(correct / total) for correct, total in zip(total_correct, total_count)]


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
            logger.info("uploading .pth model...")
            upload_file(
                path_or_fileobj=self.model_local,
                path_in_repo=self.repo_pretrained_model,
                repo_id=self.repo_id,
                repo_type="model",
            )
        except Exception:
            logger.exception("upload .pth model failed")

    def upload_onnx(self):
        try:
            logger.info("uploading .onnx model...")
            upload_file(
                path_or_fileobj=self.model_onnx,
                path_in_repo=self.repo_pretrained_model,
                repo_id=self.repo_id,
                repo_type="model",
            )
        except Exception:
            logger.exception("upload .onnx model failed")
