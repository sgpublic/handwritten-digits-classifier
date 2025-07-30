from typing import Callable

from torch import Tensor

from vision_models.core.model_trainer import ModelTrainer
from vision_models.models.cifar10.cifar10_model import Cifar10Model


class Cifar10ModelTrainer(Cifar10Model, ModelTrainer):
    @property
    def trainer_pre_transform(self) -> list[Callable[[Tensor], Tensor]]:
        return []
        
    @property
    def trainer_post_transform(self) -> list[Callable[[Tensor], Tensor]]:
        return []
