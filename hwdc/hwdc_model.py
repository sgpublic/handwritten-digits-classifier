import os.path
import shutil

import numpy
import torch
from PIL import Image, ImageOps
from huggingface_hub import hf_hub_download
from torch import nn, Tensor
from torchvision.models.resnet import _resnet, BasicBlock, ResNet

from hwdc.core.config import HWDC_DEVICE
from hwdc.core.logger import create_logger
from hwdc.core.resource import hwdc_path
from hwdc.core.tensor import images_to_batch_tenser

logger = create_logger(__name__)


class HwdcModel:
    @property
    def repo_id(self):
        return "mhmzx/handwritten-digits-classifier"

    @property
    def repo_pretrained_model(self):
        return "hwdc_pretrained.pth"

    @property
    def repo_pretrained_model(self):
        return "hwdc_pretrained.onnx"

    def __init__(self):
        self._resnet_model = self._create_empty_model()
        _device = HWDC_DEVICE
        if _device == "cuda" and not torch.cuda.is_available():
            logger.warn("cuda is not available, use cpu as fallback.")
            _device = "cpu"
        self._device = torch.device(_device)
        self._model_local = hwdc_path("./model/hwdc_local.pth")
        self._model_onnx = hwdc_path("./model/hwdc_local.onnx")

    @property
    def model_local(self):
        return self._model_local

    @property
    def model_onnx(self):
        return self._model_onnx

    @property
    def resnet_model(self):
        return self._resnet_model

    def move_to_device(self, obj: any):
        return obj.to(self._device)

    @staticmethod
    def _create_empty_model() -> ResNet:
        # model = torchvision.models.resnet18(
        #     num_classes=10,
        # )
        model = _resnet(
            block=BasicBlock,
            layers=[1, 1, 1, 1],
            weights=None,
            progress=True,
            num_classes=10
        )
        model.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2, bias=False)
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

            self.resnet_model.load_state_dict(state_dict)

            logger.info("load model finished")
            return True
        except FileNotFoundError:
            logger.warn("weights not exist!")
            return False
        except Exception:
            logger.exception("weights load failed!")
            return False
        finally:
            self.move_to_device(self.resnet_model)
            self.resnet_model.eval()

    def preprocess(self, images: list[Image]) -> Tensor:
        payload = [image.resize((28, 28), Image.LANCZOS) for image in images]
        payload = [ImageOps.invert(image) for image in payload]
        payload = images_to_batch_tenser(payload)
        return payload

    def predict(self, payload: Tensor) -> list[tuple[int, float]]:
        with torch.no_grad():
            payload = self.move_to_device(payload)
            model_output = self.resnet_model(payload)
            probs = torch.softmax(model_output, dim=1)
            values, indices = torch.max(probs, dim=1)
            return list(zip(indices.tolist(), values.tolist()))

    def export_as_onnx(self):
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 28, 28)
            torch.onnx.export(
                self.resnet_model,
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
