from typing import Callable

from PIL.Image import Image
from torch import Tensor
from torchvision.models import ResNet, VGG

from vision_models.core.model import Model


class Cifar10Model(Model):
    def _create_empty_resnet_custom_model(self) -> ResNet:
        pass

    def _create_empty_vgg_custom_model(self) -> VGG:
        pass

    def save_weight(self):
        pass

    @property
    def pre_transform(self) -> list[Callable[[Image], Image]]:
        return []

    @property
    def post_transform(self) -> list[Callable[[Tensor], Tensor]]:
        return []

