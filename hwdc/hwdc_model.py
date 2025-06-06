import os.path
import shutil

import torch
from huggingface_hub import hf_hub_download
from torch import nn
from torchvision.models.resnet import _resnet, BasicBlock, ResNet

from hwdc.core.config import HWDC_DEVICE
from hwdc.core.logger import create_logger
from hwdc.core.resource import hwdc_path

logger = create_logger(__name__)


class HwdcModel:
    _repo_id = "mhmzx/handwritten-digits-classifier"
    _repo_pretrained_model = "hwdc_pretrained.pth"

    def __init__(self,
                 use_pretrained: bool = True):
        self._use_pretrained = use_pretrained
        self._resnet_model = self._create_empty_model()
        _device = HWDC_DEVICE
        if _device == "cuda" and not torch.cuda.is_available():
            logger.warn("cuda is not available, use cpu as fallback.")
            _device = "cpu"
        self._device = torch.device(_device)
        self._model_local = hwdc_path("./model/hwdc_local.pth")
        self._model_pretrained = None

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

    def load(self) -> bool:
        logger.info("load model weights...")
        try:
            if self._use_pretrained:
                model_pretrained = hf_hub_download(
                    repo_id=self._repo_id,
                    filename=self._repo_pretrained_model,
                )
                state_dict = torch.load(model_pretrained)
            else:
                if not os.path.exists(self._model_local) or not os.path.isfile(self._model_local):
                    raise FileNotFoundError("weights not exist!")
                shutil.copy(self._model_local, f"{self._model_local}.bak")
                state_dict = torch.load(self._model_local)

            self._resnet_model.load_state_dict(state_dict)

            logger.info("load model finished")
            return True
        except FileNotFoundError:
            logger.warn("weights not exist!")
            return False
        except Exception:
            logger.exception("weights load failed!")
            return False
        finally:
            self._resnet_model.to(self._device)
