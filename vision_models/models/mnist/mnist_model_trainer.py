from typing import Callable

import torch
from torch import Tensor
from torchvision import transforms

from vision_models.core.model_trainer import VisionClassifyModelTrainer, DatasetConfig
from vision_models.core.types.dataset_type import DatasetSplitType, DatasetColumnType
from vision_models.core.types.model_save_type import ModelSaveType
from vision_models.models.mnist.config import MNIST_DATASET_RANDOM_ROTATE, MNIST_DATASET_RANDOM_SCALE, \
    MNIST_DATASET_RANDOM_ELASTIC_ALPHA, MNIST_DATASET_RANDOM_ELASTIC_SIGMA
from vision_models.models.mnist.mnist_model import MnistModel


class MnistModelTrainer(MnistModel, VisionClassifyModelTrainer):
    @property
    def __logger_name__(self) -> str:
        return __name__

    @property
    def dataset_config(self) -> DatasetConfig:
        return DatasetConfig(
            # https://huggingface.co/datasets/ylecun/mnist
            path="ylecun/mnist",
            columns={
                DatasetColumnType.IMAGE: "image",
                DatasetColumnType.LABEL: "label",
            },
            splits={
                DatasetSplitType.TRAIN: "train",
                DatasetSplitType.TEST: "test",
            },
        )

    def _save_as_onnx(self):
        torch.onnx.export(
            self.model,
            self.move_to_device(torch.randn(1, 1, 28, 28)),
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
        random_rotate = MNIST_DATASET_RANDOM_ROTATE
        random_scale = MNIST_DATASET_RANDOM_SCALE
        random_elastic_alpha = MNIST_DATASET_RANDOM_ELASTIC_ALPHA
        random_elastic_sigma = MNIST_DATASET_RANDOM_ELASTIC_SIGMA
        return [
            # 加入随机缩放和旋转
            transforms.RandomAffine(degrees=(0 - random_rotate, 0 + random_rotate),
                                    scale=(1.0 - random_scale, 1.0 + random_scale)),
            # 加入随机变形
            transforms.ElasticTransform(alpha=random_elastic_alpha, sigma=random_elastic_sigma),
        ]
