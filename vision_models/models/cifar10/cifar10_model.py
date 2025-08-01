from typing import Callable

from PIL.Image import Image
from torch import Tensor, nn
from torchvision import models
from torchvision.models import ResNet

from vision_models.core.model import VisionClassifyModel


class Cifar10Model(VisionClassifyModel):
    @property
    def __logger_name__(self) -> str:
        return __name__

    def _create_empty_resnet_custom_model(self) -> ResNet:
        model = models.resnet34(
            num_classes=10,
        )
        # 更改输入通道为 1、换用 3x3 大小的卷积核
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # 换用 LeakyReLU 以解决 ReLU 的神经元死亡问题
        model.relu = nn.LeakyReLU(inplace=True)
        # 取消池化，把池化核改为 1x1，步长为 1，填充 0，这样就能实现无池化的效果
        model.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        return model

    @property
    def pre_transform(self) -> list[Callable[[Image], Image]]:
        return []

    @property
    def post_transform(self) -> list[Callable[[Tensor], Tensor]]:
        return []
