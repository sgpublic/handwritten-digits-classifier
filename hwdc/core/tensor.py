from typing import Dict

import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms

_transform = transforms.ToTensor()


def image_to_tenser(image: Image) -> Tensor:
    return _transform(image.convert("L"))


def image_to_batch_tenser(image: Image) -> Tensor:
    return image_to_tenser(image).unsqueeze(0)


def batch_to_tensor(batch: any) -> dict[str, Tensor]:
    inputs = [image_to_tenser(img) for img in batch["image"]]
    labels = torch.tensor(batch["label"])
    return {
        "image": torch.stack(inputs),
        "label": labels,
    }
