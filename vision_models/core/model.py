import os.path
import shutil
from abc import abstractmethod, ABC
from logging import Logger
from typing import Callable, Optional

import torch
from PIL.Image import Image
from huggingface_hub import hf_hub_download
from torch import Tensor
from torch.nn import Module
from torchvision.models import ResNet, VGG

from vision_models.core.config import CORE_DEVICE, CORE_DATASET_TYPE, CORE_MODEL_TYPE, CORE_DEBUG
from vision_models.core.log import Log
from vision_models.core.types.model_save_type import ModelSaveType
from vision_models.core.types.model_type import ModelType
from vision_models.core.utils.resource import resource_path
from vision_models.core.utils.tensor import images_to_batch_tenser


class VisionClassifyModel(Log, ABC):
    @property
    def __logger_name__(self) -> str:
        return __name__

    def __init__(self):
        super().__init__()
        _device = CORE_DEVICE
        if _device == "cuda" and not torch.cuda.is_available():
            self.logger.warn("cuda is not available, use cpu as fallback.")
            _device = "cpu"
        self._device = torch.device(_device)

        self._dataset_type = CORE_DATASET_TYPE
        self._model = self._create_empty_model()

    @property
    def model_type(self) -> ModelType:
        return CORE_MODEL_TYPE

    @property
    def repo_id(self) -> str:
        return "mhmzx/vision-models"

    @property
    def is_debug(self) -> bool:
        return CORE_DEBUG

    def repo_pretrained_model(self, model_type: ModelSaveType) -> str:
        filename = model_type.with_file_name("model_weight")
        return f"{self.model_base_path}/{filename}"

    @property
    def model_base_path(self) -> str:
        return f"{self._dataset_type.value}/{self.model_type.value}"

    def model_local(self, model_type: ModelSaveType) -> str:
        filename = model_type.with_file_name("model_weight")
        return resource_path(f"./model_save/{self.model_base_path}/{filename}")

    @property
    def model(self) -> Module:
        return self._model

    def move_to_device(self, obj: any):
        return obj.to(self._device)

    def _create_empty_model(self) -> Module:
        model: Optional[Module] = None
        match self.model_type:
            case ModelType.ResNet_Custom:
                model = self._create_empty_resnet_custom_model()
            case ModelType.ResNetV2_Custom:
                model = self._create_empty_resnet_v2_custom_model()
            case ModelType.VGG_Custom:
                model = self._create_empty_vgg_custom_model()
        if model is None:
            raise ValueError(f"model_type {self.model_type} not supported!")
        return model

    # noinspection PyMethodMayBeStatic
    def _create_empty_resnet_custom_model(self) -> Optional[ResNet]:
        return None

    # noinspection PyMethodMayBeStatic
    def _create_empty_resnet_v2_custom_model(self) -> Optional[ResNet]:
        return None

    # noinspection PyMethodMayBeStatic
    def _create_empty_vgg_custom_model(self) -> Optional[VGG]:
        return None

    def load_weight(self, use_pretrained: bool = True) -> bool:
        try:
            self.logger.info("load model weights...")
            if use_pretrained:
                self.logger.info("downloading model weight...")
                model_pretrained = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=self.repo_pretrained_model(model_type=ModelSaveType.ORIGIN),
                )
                self.logger.info("download model weight finished")
                state_dict = torch.load(model_pretrained)
            else:
                self.logger.info("prepare local model weights...")
                model_path = self.model_local(model_type=ModelSaveType.ORIGIN)
                if not os.path.exists(model_path) or not os.path.isfile(model_path):
                    raise FileNotFoundError("weights not exist!")
                shutil.copy(model_path, f"{model_path}.bak")
                self.logger.info("prepare local model weights finished")
                state_dict = torch.load(model_path)

            self.model.load_state_dict(state_dict)

            self.logger.info("load model weights finished")
            return True
        except FileNotFoundError:
            self.logger.warn("weights not exist!")
            return False
        except Exception:
            self.logger.exception("weights load failed!")
            return False
        finally:
            self.move_to_device(self.model)
            self.model.eval()

    @property
    @abstractmethod
    def pre_transform(self) -> list[Callable[[Image], Image]]:
        return []

    @property
    @abstractmethod
    def post_transform(self) -> list[Callable[[Tensor], Tensor]]:
        return []

    def preprocess(self, images: list[Image]) -> Tensor:
        payload = images_to_batch_tenser(images, pre_transform=self.pre_transform, post_transform=self.post_transform)
        return payload

    def predict(self, payload: Tensor) -> list[tuple[int, float]]:
        with torch.no_grad():
            payload = self.move_to_device(payload)
            model_output = self.model(payload)
            probs = torch.softmax(model_output, dim=1)
            values, indices = torch.max(probs, dim=1)
            return list(zip(indices.tolist(), values.tolist()))
