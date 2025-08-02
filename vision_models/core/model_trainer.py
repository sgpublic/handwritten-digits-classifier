import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy
import pandas
import torch
from PIL.Image import Image
from huggingface_hub import upload_file
from matplotlib import pyplot as plt
from torch import nn, optim, Tensor
from torch.nn.modules.loss import _WeightedLoss
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT
from torch.utils.data import DataLoader

from vision_models.core.types.dataset_type import DatasetColumnType, DatasetSplitType
from vision_models.core.types.model_save_type import ModelSaveType
from vision_models.core.utils.resource import resource_path
from vision_models.core.utils.strings import arr2str, float2str
from vision_models.core.model import VisionClassifyModel
from vision_models.core.utils.dataset_loader import data_loader
from vision_models.trainer.config import TRAINER_DATASET_MAX_EPOCHS, TRAINER_DATASET_BATCH_SIZE, \
    TRAINER_DATASET_TEST_DATASET_SIZE, \
    TRAINER_MODEL_ACCURACY_THRESHOLD, TRAINER_LEARN_RATE, TRAINER_CHART_SAVE_INDICATORS


@dataclass
class DatasetConfig:
    path: str
    columns: dict[DatasetColumnType, str]
    splits: dict[DatasetSplitType, str]

class VisionClassifyModelTrainer(VisionClassifyModel, ABC):
    @property
    def __logger_name__(self) -> str:
        return __name__

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def dataset_config(self) -> DatasetConfig:
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

    # noinspection PyMethodMayBeStatic
    def create_criterion(self):
        return nn.CrossEntropyLoss()

    def create_optimizer(self, parameters: ParamsT, learn_rate: float):
        return optim.Adam(self.model.parameters(), lr=learn_rate)

    def train(self,
              epochs: int = TRAINER_DATASET_MAX_EPOCHS,
              batch_size: int = TRAINER_DATASET_BATCH_SIZE,
              test_dataset_size: int = TRAINER_DATASET_TEST_DATASET_SIZE,
              accuracy_threshold: float = TRAINER_MODEL_ACCURACY_THRESHOLD,
    ):
        self.logger.info("prepare training")
        train_loader = data_loader(
            path=self.dataset_config.path,
            split=self.dataset_config.splits[DatasetSplitType.TRAIN],
            columns=self.dataset_config.columns,
            batch_size=batch_size,
            pre_transform=self.pre_transform + self.trainer_pre_transform,
            post_transform=self.post_transform + self.trainer_post_transform,
        )

        criterion = self.create_criterion()
        optimizer = self.create_optimizer(parameters=self.model.parameters(), learn_rate=TRAINER_LEARN_RATE)

        self.logger.info("start training")
        self.model.train()
        best_loss = None
        loss_list: Optional[list[float]] = None
        avg_loss_list: Optional[list[float]] = None
        steps_list: Optional[list[int]] = None
        avg_accuracy_list: Optional[list[float]] = None
        accuracy_list: Optional[list[list[float]]] = None
        if TRAINER_CHART_SAVE_INDICATORS:
            loss_list = []
            avg_loss_list = []
            steps_list = []
            avg_accuracy_list = []
            accuracy_list = [[] for _ in range(self.num_classes)]

        for index in range(epochs):
            self.logger.info(f"------------  epoch [{index + 1}/{epochs}] start  ------------")
            self.logger.info("training...")
            steps, avg_loss = self._real_train(
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                loss_list=loss_list
            )
            self.logger.info(f"loss: {avg_loss:.6f}")
            self.logger.info("testing...")
            avg_accuracy, current_accuracy = self._test(test_size=test_dataset_size)
            self.logger.info(f"accuracy({float2str(avg_accuracy)} in average): {arr2str(current_accuracy)}")
            if TRAINER_CHART_SAVE_INDICATORS:
                if len(steps_list) == 0:
                    steps_list.append(steps)
                else:
                    steps_list.append(steps_list[len(steps_list) - 1] + steps)
                avg_loss_list.append(avg_loss)
                avg_accuracy_list.append(avg_accuracy)
                for i in range(len(current_accuracy)):
                    accuracy_list[i].append(current_accuracy[i])
                self.save_chart(
                    loss_list=loss_list,
                    avg_loss_list=avg_loss_list,
                    steps_list=steps_list,
                    avg_accuracy_list=avg_accuracy_list,
                    accuracy_list=accuracy_list
                )
            self.logger.info(f"-------------  epoch [{index + 1}/{epochs}] end  -------------")


            if best_loss is None:
                best_loss = avg_loss
            if best_loss < avg_loss:
                self.logger.info("loss increased, skip saving model weights")
                continue
            self.logger.info("loss decreased, save model weights")
            if not self.save():
                self.logger.exception("failed to save model weights, stop training")
                break

            # 要求每个数字的准确率都达到预期
            # if all(x >= accuracy_threshold for x in current_accuracy):
            # 要求平均准确率达到预期
            if avg_accuracy >= accuracy_threshold:
                self.logger.info("the accuracy requirement is met, stop training")
                break
        self.logger.info("training finished")

    def _real_train(self, train_loader: DataLoader, criterion: _WeightedLoss, optimizer: Optimizer, loss_list: Optional[list[float]]) -> tuple[int, float]:
        total_loss: float = 0
        current_step: int = 0
        for batch in train_loader:
            inputs, labels = batch[DatasetColumnType.IMAGE], batch[DatasetColumnType.LABEL]
            inputs, labels = self.move_to_device(inputs), self.move_to_device(labels)

            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            current_step += 1
            if loss_list is not None:
                loss_list.append(loss.item())

        return current_step, total_loss / len(train_loader)

    def _test(self, test_size: int = 10000) -> tuple[float, list[float]]:
        with torch.no_grad():
            total_count = numpy.zeros(self.num_classes, dtype=numpy.int32)
            total_correct = numpy.zeros(self.num_classes, dtype=numpy.int32)
            test_loader = data_loader(
                path=self.dataset_config.path,
                split=self.dataset_config.splits[DatasetSplitType.TEST],
                columns=self.dataset_config.columns,
                batch_size=test_size,
                dataset_size=test_size
            )
            # 实际上这个循环只跑一次
            for batch in test_loader:
                inputs, labels = batch[DatasetColumnType.IMAGE], batch[DatasetColumnType.LABEL].tolist()
                outputs = self.predict(inputs)
                for (a, _), b in zip(outputs, labels):
                    total_count[b] += 1
                    if a == b:
                        total_correct[a] += 1
            return float(total_correct.sum() / total_count.sum()), [float(correct / total) for correct, total in zip(total_correct, total_count)]

    def save_chart(self,
                   loss_list: list[float],
                   avg_loss_list: list[float],
                   steps_list: list[int],
                   avg_accuracy_list: list[float],
                   accuracy_list: list[list[float]]) -> bool:
        if not TRAINER_CHART_SAVE_INDICATORS:
            self.logger.info("skip save charts")
            return False
        try:
            self.logger.info("save charts...")
            self._save_line_chart(
                title=f"Loss per Step",
                x_label="step", x=numpy.arange(len(loss_list)),
                y_label="loss", y=loss_list,
                save_name="loss"
            )
            self._save_line_chart(
                title=f"Average Loss",
                x_label="step", x=steps_list,
                y_label="avg_loss", y=avg_loss_list,
                save_name="avg_loss"
            )
            self._save_line_chart(
                title=f"Average Accuracy",
                x_label="step", x=steps_list,
                y_label="avg_accuracy", y=avg_accuracy_list, y_ticks=numpy.linspace(0.0, 1.0, 6),
                save_name="avg_accuracy"
            )
            for accuracy_index in range(len(accuracy_list)):
                accuracy_item = accuracy_list[accuracy_index]
                self._save_line_chart(
                    title=f"Accuracy of Class {accuracy_index}",
                    x_label="step", x=steps_list,
                    y_label="accuracy", y=accuracy_item, y_ticks=numpy.linspace(0.0, 1.0, 6),
                    save_name=f"accuracy-class_{accuracy_index}"
                )
            self.logger.info("save charts finished")
            return True
        except Exception:
            self.logger.exception("failed to save charts")
            return False

    @property
    def charts_base_path(self):
        return resource_path("./charts", self.save_base_path)

    def _save_line_chart(self,
                         title: str,
                         x_label: str, x: Union[list[int], numpy.ndarray],
                         y_label: str, y: Union[list[float], numpy.ndarray],
                         save_name: str,
                         x_ticks: Optional[Union[range, numpy.ndarray]] = None,
                         y_ticks: Optional[Union[range, numpy.ndarray]] = None,
                         figsize: tuple[float, float] = (15, 6), dpi: int = 300):
        os.makedirs(self.charts_base_path, exist_ok=True)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        if type(x) == list:
            x = numpy.array(x)
        if type(y) == list:
            y = numpy.array(y)
        loss_data = pandas.DataFrame({
            x_label: x,
            y_label: y,
        })
        loss_data.plot.line(x=x_label, y=y_label, ax=ax)
        ax.set_title(title)
        if x_ticks is not None:
            ax.set_xticks(x_ticks)
        if y_ticks is not None:
            ax.set_yticks(y_ticks)
        fig.savefig(resource_path(f"{save_name}.png", self.charts_base_path))
        plt.close(fig)

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
            self.logger.exception("weight save failed!")
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
            self.logger.info(f"uploading .{model_type.value} model weights...")
            upload_file(
                path_or_fileobj=self.model_local(model_type=model_type),
                path_in_repo=self.repo_pretrained_model(model_type=model_type),
                repo_id=self.repo_id,
                repo_type="model",
            )
        except Exception:
            self.logger.exception(f"upload .{model_type.value} model weights failed")
