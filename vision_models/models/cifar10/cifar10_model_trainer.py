from typing import Callable

import torch
from torch import Tensor, optim
from torch.optim.optimizer import ParamsT

from vision_models.core.model_trainer import VisionClassifyModelTrainer, DatasetConfig
from vision_models.core.types.dataset_type import DatasetColumnType, DatasetSplitType
from vision_models.core.types.model_save_type import ModelSaveType
from vision_models.models.cifar10.cifar10_model import Cifar10Model
from vision_models.models.cifar10.config import CIFAR10_RESNET_WEIGHT_DECAY


class Cifar10ModelTrainer(Cifar10Model, VisionClassifyModelTrainer):
    @property
    def __logger_name__(self) -> str:
        return __name__

    @property
    def dataset_config(self) -> DatasetConfig:
        return DatasetConfig(
            # https://huggingface.co/datasets/uoft-cs/cifar10
            path="uoft-cs/cifar10",
            columns={
                DatasetColumnType.IMAGE: "img",
                DatasetColumnType.LABEL: "label",
            },
            splits={
                DatasetSplitType.TRAIN: "train",
                DatasetSplitType.TEST: "test",
            },
        )

    def create_optimizer(self, parameters: ParamsT, learn_rate: float):
        # 设置 weight_decay 以应对过拟合
        return optim.Adam(params=parameters, lr=learn_rate, weight_decay=CIFAR10_RESNET_WEIGHT_DECAY)

    def _save_as_onnx(self):
        torch.onnx.export(
            self.model,
            self.move_to_device(torch.randn(1, 3, 32, 32)),
            self.model_local(model_type=ModelSaveType.ONNX),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            opset_version=12,
            verbose=self.is_debug,
        )

    @property
    def trainer_pre_transform(self) -> list[Callable[[Tensor], Tensor]]:
        return []
        
    @property
    def trainer_post_transform(self) -> list[Callable[[Tensor], Tensor]]:
        return []
