from typing import Callable

import torch
from torch import Tensor

from vision_models.core.model_trainer import VisionClassifyModelTrainer
from vision_models.core.types.model_save_type import ModelSaveType
from vision_models.models.cifar10.cifar10_model import Cifar10Model


class Cifar10ModelTrainer(Cifar10Model, VisionClassifyModelTrainer):
    def _save_as_onnx(self):
        torch.onnx.export(
            self.model,
            self.move_to_device(torch.randn(1, 3, 28, 28)),
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

    def dataset_path(self) -> str:
        return "uoft-cs/cifar10"

    @property
    def trainer_pre_transform(self) -> list[Callable[[Tensor], Tensor]]:
        return []
        
    @property
    def trainer_post_transform(self) -> list[Callable[[Tensor], Tensor]]:
        return []
