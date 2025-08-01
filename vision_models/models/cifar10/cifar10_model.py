from typing import Callable

from PIL.Image import Image
from torch import Tensor
from torchvision import models
from torchvision.models import ResNet

from vision_models.core.model import VisionClassifyModel


class Cifar10Model(VisionClassifyModel):
    def _create_empty_resnet_custom_model(self) -> ResNet:
        model = models.resnet34(
            num_classes=10,
        )
        return model

    @property
    def pre_transform(self) -> list[Callable[[Image], Image]]:
        return []

    @property
    def post_transform(self) -> list[Callable[[Tensor], Tensor]]:
        return []
