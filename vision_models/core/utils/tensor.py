from typing import Callable, Optional

import torch
from PIL.Image import Image
from torch import Tensor
from torchvision import transforms

from vision_models.core.types.dataset_type import DatasetColumnType


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
    columns: dict[DatasetColumnType, str],
    pre_transform: Optional[list[Callable[[Image], Image]]] = None,
    post_transform: Optional[list[Callable[[Tensor], Tensor]]] = None,
) -> dict[DatasetColumnType, Tensor]:
    if post_transform is None:
        post_transform = []
    if pre_transform is None:
        pre_transform = []
    inputs = images_to_batch_tenser(batch[columns[DatasetColumnType.IMAGE]], pre_transform=pre_transform, post_transform=post_transform)
    labels = torch.tensor(batch[columns[DatasetColumnType.LABEL]])
    return {
        DatasetColumnType.IMAGE: inputs,
        DatasetColumnType.LABEL: labels,
    }
