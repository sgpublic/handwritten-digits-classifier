from typing import Callable, Optional

import torch
from PIL.Image import Image
from torch import Tensor
from torchvision import transforms

def image_to_tenser(
        image: Image,
        pre_transform: Optional[list[Callable[[Image], Image]]] = None,
        post_transform: Optional[list[Callable[[Tensor], Tensor]]] = None,
) -> Tensor:
    if post_transform is None:
        post_transform = []
    if pre_transform is None:
        pre_transform = []
    transform = transforms.Compose(pre_transform + [transforms.ToTensor()] + post_transform)
    return transform(image)

def images_to_batch_tenser(
        images: list[Image],
        pre_transform: Optional[list[Callable[[Image], Image]]] = None,
        post_transform: Optional[list[Callable[[Tensor], Tensor]]] = None,
) -> Tensor:
    if post_transform is None:
        post_transform = []
    if pre_transform is None:
        pre_transform = []
    return torch.stack([image_to_tenser(image, pre_transform=pre_transform, post_transform=post_transform) for image in images])

def batch_to_tensor(
        batch: dict,
        pre_transform: Optional[list[Callable[[Image], Image]]] = None,
        post_transform: Optional[list[Callable[[Tensor], Tensor]]] = None,
) -> dict[str, Tensor]:
    if post_transform is None:
        post_transform = []
    if pre_transform is None:
        pre_transform = []
    inputs = images_to_batch_tenser(batch["image"], pre_transform=pre_transform, post_transform=post_transform)
    labels = torch.tensor(batch["label"])
    return {
        "image": inputs,
        "label": labels,
    }
