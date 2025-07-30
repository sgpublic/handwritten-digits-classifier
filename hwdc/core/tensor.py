from typing import Callable, Optional

import torch
from PIL.Image import Image
from torch import Tensor
from torchvision import transforms

def image_to_tenser(
        image: Image,
        pre_transform: Optional[list[Callable[[Image], Tensor]]] = None,
        post_transform: Optional[list[Callable[[Image], Tensor]]] = None,
) -> Tensor:
    if post_transform is None:
        post_transform = []
    if pre_transform is None:
        pre_transform = []
    transform = transforms.Compose(pre_transform + [transforms.ToTensor()] + post_transform)
    return transform(image.convert("L"))

def images_to_batch_tenser(
        images: list[Image],
        pre_transform: Optional[list[Callable[[Image], Tensor]]] = None,
        post_transform: Optional[list[Callable[[Image], Tensor]]] = None,
) -> Tensor:
    if post_transform is None:
        post_transform = []
    if pre_transform is None:
        pre_transform = []
    return torch.stack([image_to_tenser(image, post_transform, pre_transform) for image in images])

def batch_to_tensor(
        batch: dict,
        pre_transform: Optional[list[Callable[[Image], Tensor]]] = None,
        post_transform: Optional[list[Callable[[Image], Tensor]]] = None,
) -> Tensor:
    if post_transform is None:
        post_transform = []
    if pre_transform is None:
        pre_transform = []
    inputs = images_to_batch_tenser(batch["image"], post_transform, pre_transform)
    labels = torch.tensor(batch["label"])
    return {
        "image": inputs,
        "label": labels,
    }
