import random
from typing import Optional, Callable

from PIL.Image import Image
from datasets import DownloadConfig, load_dataset
from torch import Tensor
from torch.utils.data import DataLoader
from datasets.arrow_dataset import Dataset

from vision_models.core.utils.logger import create_logger
from vision_models.core.utils.tensor import batch_to_tensor

logger = create_logger(__name__)

def mnist_dataset(
    split: str = "train",
    dataset_size: int = None,
) -> Dataset:
    dataset = load_dataset(
        path="mnist",
        download_config=DownloadConfig(
            resume_download=True,
            force_download=False,
        ),
    )[split]
    if dataset_size is not None:
        indices = random.sample(range(len(dataset)), dataset_size)
        dataset = dataset.select(indices)
    return dataset

# https://huggingface.co/datasets/ylecun/mnist
# https://huggingface.co/docs/datasets/use_with_pytorch#stream-data
def mnist_dataset_loader(
        split: str = "train",
        batch_size: int = 1000,
        dataset_size: Optional[int] = None,
        pre_transform: Optional[list[Callable[[Image], Image]]] = None,
        post_transform: Optional[list[Callable[[Tensor], Tensor]]] = None,
) -> DataLoader:
    logger.info(f"loading mnist dataset[{split}]...")
    dataset = mnist_dataset(
        split=split,
        dataset_size=dataset_size
    )
    dataset = dataset.with_transform(
        transform=lambda x: batch_to_tensor(batch=x, pre_transform=pre_transform, post_transform=post_transform),
        columns=["image", "label"],
    )
    logger.info(f"load mnist dataset[{split}] finished")
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
    )
