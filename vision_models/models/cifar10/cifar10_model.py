from typing import Callable

from PIL.Image import Image
from torch import Tensor, nn
from torchvision import models
from torchvision.models import ResNet

from vision_models.core.model import VisionClassifyModel
from vision_models.models.cifar10.config import CIFAR10_RESNET_DROPOUT


class Cifar10Model(VisionClassifyModel):
    @property
    def __logger_name__(self) -> str:
        return __name__

    @property
    def num_classes(self) -> int:
        return 10

    def _create_empty_resnet_custom_model(self) -> ResNet:
        model = models.resnet34(
            num_classes=self.num_classes
        )
        # 换用 LeakyReLU 以解决 ReLU 的神经元死亡问题
        model.relu = nn.LeakyReLU(inplace=True)
        return model

    @property
    def pre_transform(self) -> list[Callable[[Image], Image]]:
        return []

    @property
    def post_transform(self) -> list[Callable[[Tensor], Tensor]]:
        return []
