import os.path
import shutil

import torch
from PIL import Image, ImageOps
from huggingface_hub import hf_hub_download
from torch import nn, Tensor
from torch.nn import Module
from torchvision.models.resnet import _resnet, BasicBlock, ResNet

from hwdc.core.config import HWDC_DEVICE, HWDC_MODEL_TYPE
from hwdc.core.logger import create_logger
from hwdc.core.resource import hwdc_path
from hwdc.core.tensor import images_to_batch_tenser
from hwdc.hwdc_model_type import HwdcModelType

logger = create_logger(__name__)


class HwdcModel:
    @property
    def repo_id(self):
        return "mhmzx/handwritten-digits-classifier"

    @property
    def repo_pretrained_model(self):
        return f"{self._model_type.value}/hwdc_pretrained.pth"

    @property
    def repo_pretrained_model(self):
        return f"{self._model_type.value}/hwdc_pretrained.onnx"

    def __init__(self):
        self._model_type = HWDC_MODEL_TYPE
        self._model = self._create_empty_model(self._model_type)
        _device = HWDC_DEVICE
        if _device == "cuda" and not torch.cuda.is_available():
            logger.warn("cuda is not available, use cpu as fallback.")
            _device = "cpu"
        self._device = torch.device(_device)
        self._model_base_path = hwdc_path(f"./model/{self._model_type.value}")
        self._model_local = hwdc_path("./hwdc_local.pth", self._model_base_path)
        self._model_onnx = hwdc_path("./hwdc_local.onnx", self._model_base_path)

    @property
    def model_local(self):
        return self._model_local

    @property
    def model_onnx(self):
        return self._model_onnx

    @property
    def model(self) -> Module:
        return self._model

    @property
    def model_type(self) -> HwdcModelType:
        return self._model_type

    def move_to_device(self, obj: any):
        return obj.to(self._device)

    @staticmethod
    def _create_empty_model(model_type: HwdcModelType) -> Module:
        logger.info(f"create model of type {model_type.name}")
        if model_type == HwdcModelType.ResNet:
            return HwdcModel._create_empty_resnet_model()
        return None

    @staticmethod
    def _create_empty_resnet_model() -> ResNet:
        model = _resnet(
            block=BasicBlock,
            layers=[1, 1, 1, 1],
            weights=None,
            progress=True,
            num_classes=10
        )
        model.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2, bias=False)
        # 换用 LeakyReLU 以解决 ReLU 的神经元死亡问题
        model.relu = nn.LeakyReLU(inplace=True)
        return model

    def load(self, use_pretrained: bool = True) -> bool:
        logger.info("load model weights...")
        try:
            if use_pretrained:
                logger.info("downloading model weight...")
                model_pretrained = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=self.repo_pretrained_model,
                )
                logger.info("download model weight finished")
                state_dict = torch.load(model_pretrained)
            else:
                if not os.path.exists(self.model_local) or not os.path.isfile(self.model_local):
                    raise FileNotFoundError("weights not exist!")
                shutil.copy(self.model_local, f"{self.model_local}.bak")
                state_dict = torch.load(self.model_local)

            self.model.load_state_dict(state_dict)

            logger.info("load model finished")
            return True
        except FileNotFoundError:
            logger.warn("weights not exist!")
            return False
        except Exception:
            logger.exception("weights load failed!")
            return False
        finally:
            self.move_to_device(self.model)
            self.model.eval()

    def preprocess(self, images: list[Image]) -> Tensor:
        payload = [image.resize((28, 28), Image.LANCZOS) for image in images]
        payload = [ImageOps.invert(image) for image in payload]
        payload = images_to_batch_tenser(payload)
        return payload

    def predict(self, payload: Tensor) -> list[tuple[int, float]]:
        with torch.no_grad():
            payload = self.move_to_device(payload)
            model_output = self.model(payload)
            probs = torch.softmax(model_output, dim=1)
            values, indices = torch.max(probs, dim=1)
            return list(zip(indices.tolist(), values.tolist()))

    def export_as_onnx(self):
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 28, 28)
            torch.onnx.export(
                self.model,
                dummy_input,
                self.model_onnx,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"},
                },
                opset_version=12,
                verbose=True,
            )
