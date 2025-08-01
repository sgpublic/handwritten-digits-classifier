import os
from abc import ABC, abstractmethod
from typing import Callable

import numpy
import torch
from PIL.Image import Image
from huggingface_hub import upload_file
from torch import nn, optim, Tensor
from torch.nn.modules.loss import _WeightedLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from vision_models.core.types.model_save_type import ModelSaveType
from vision_models.core.utils.logger import create_logger
from vision_models.core.utils.strings import arr2str, float2str
from vision_models.core.model import VisionClassifyModel
from vision_models.core.utils.dataset_loader import data_loader
from vision_models.trainer.config import TRAINER_DATASET_MAX_EPOCHS, TRAINER_DATASET_BATCH_SIZE, TRAINER_DATASET_TEST_DATASET_SIZE, \
    TRAINER_MODEL_ACCURACY_THRESHOLD, TRAINER_LEARN_RATE

logger = create_logger(__name__)


class VisionClassifyModelTrainer(VisionClassifyModel, ABC):
    @property
    @abstractmethod
    def dataset_path(self) -> str:
        pass

    @property
    @abstractmethod
    def trainer_pre_transform(self) -> list[Callable[[Image], Image]]:
        return []

    @property
    @abstractmethod
    def trainer_post_transform(self) -> list[Callable[[Tensor], Tensor]]:
        return []

    def load_weight(self, use_pretrained: bool = True) -> bool:
        return super().load_weight(use_pretrained=False)

    def train(self,
              epochs: int = TRAINER_DATASET_MAX_EPOCHS,
              batch_size: int = TRAINER_DATASET_BATCH_SIZE,
              learn_rate: float = TRAINER_LEARN_RATE,
              test_dataset_size: int = TRAINER_DATASET_TEST_DATASET_SIZE,
              accuracy_threshold: float = TRAINER_MODEL_ACCURACY_THRESHOLD,
    ):
        logger.info("prepare training")
        train_loader = data_loader(
            path=self.dataset_path,
            split="train",
            batch_size=batch_size,
            pre_transform=self.pre_transform + self.trainer_pre_transform,
            post_transform=self.post_transform + self.trainer_post_transform,
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learn_rate)

        logger.info("start training")
        self.model.train()
        best_loss = None
        for index in range(epochs):
            logger.info(f"epoch [{index + 1}/{epochs}] start...")
            avg_loss = self._real_train(train_loader=train_loader, criterion=criterion, optimizer=optimizer)
            logger.info(f"epoch [{index + 1}/{epochs}], loss: {avg_loss:.6f}")
            logger.info("testing...")
            avg_accuracy, current_accuracy = self._test(test_size=test_dataset_size)
            logger.info(f"epoch [{index + 1}/{epochs}], accuracy({float2str(avg_accuracy)} in average): {arr2str(current_accuracy)}")

            if best_loss is None:
                best_loss = avg_loss
            if best_loss < avg_loss:
                logger.info("loss increased, skip saving model_save weight")
                continue
            logger.info("loss decreased, save model_save weight")
            if not self.save():
                logger.exception("failed to save weight, stop training")
                break

            # 要求每个数字的准确率都达到预期
            # if all(x >= accuracy_threshold for x in current_accuracy):
            # 要求平均准确率达到预期
            if avg_accuracy >= accuracy_threshold:
                logger.info("the accuracy requirement is met, stop training")
                break
        logger.info("training finished")

    def _real_train(self, train_loader: DataLoader, criterion: _WeightedLoss, optimizer: Optimizer) -> float:
        total_loss = 0
        for batch in train_loader:
            inputs, labels = batch["image"], batch["label"]
            inputs, labels = self.move_to_device(inputs), self.move_to_device(labels)

            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def _test(self, test_size: int = 10000) -> tuple[float, list[float]]:
        with torch.no_grad():
            total_count = numpy.zeros(10, dtype=numpy.int32)
            total_correct = numpy.zeros(10, dtype=numpy.int32)
            test_loader = data_loader(
                path=self.dataset_path,
                split="test",
                batch_size=test_size,
                dataset_size=test_size
            )
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
        still_training = self.model.training
        if still_training:
            self.model.eval()
        try:
            with torch.no_grad():
                self._save_as_pth()
                self._save_as_onnx()
            return True
        except Exception:
            logger.exception("weight save failed!")
            return False
        finally:
            if still_training:
                self.model.train()

    def _save_as_pth(self):
        filename = self.model_local(model_type=ModelSaveType.ORIGIN)
        os.makedirs(name=os.path.dirname(filename), exist_ok=True)
        torch.save(self.model.state_dict(), filename)

    @abstractmethod
    def _save_as_onnx(self):
        pass

    def upload_all(self):
        self.upload(model_type=ModelSaveType.ORIGIN)
        self.upload(model_type=ModelSaveType.ONNX)

    def upload(self, model_type: ModelSaveType):
        try:
            logger.info(f"uploading .{model_type.value} model_save...")
            upload_file(
                path_or_fileobj=self.model_local(model_type=model_type),
                path_in_repo=self.repo_pretrained_model(model_type=model_type),
                repo_id=self.repo_id,
                repo_type="model",
            )
        except Exception:
            logger.exception(f"upload .{model_type.value} model_save failed")
