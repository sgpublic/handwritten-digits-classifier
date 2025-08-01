import random
from typing import Optional, Callable

from PIL.Image import Image
from datasets import DownloadConfig, load_dataset
from torch import Tensor
from torch.utils.data import DataLoader
from datasets.arrow_dataset import Dataset

from vision_models.core.types.dataset_type import DatasetColumnType
from vision_models.core.utils.logger import create_logger
from vision_models.core.utils.tensor import batch_to_tensor

logger = create_logger(__name__)

def _load_dataset(
    path: str,
    split: str = "train",
    dataset_size: Optional[int] = None,
) -> Dataset:
    dataset = load_dataset(
        path=path,
        download_config=DownloadConfig(
            resume_download=True,
            force_download=False,
        ),
        split=split,
    )
    if dataset_size is not None:
        indices = random.sample(range(len(dataset)), dataset_size)
        dataset = dataset.select(indices)
    return dataset

# https://huggingface.co/docs/datasets/use_with_pytorch#stream-data
def data_loader(
    path: str,
    split: str,
    columns: dict[DatasetColumnType, str],
    batch_size: int = 1000,
    dataset_size: Optional[int] = None,
    pre_transform: Optional[list[Callable[[Image], Image]]] = None,
    post_transform: Optional[list[Callable[[Tensor], Tensor]]] = None,
) -> DataLoader:
    logger.info(f"loading {path}[{split}]...")
    dataset = _load_dataset(
        path=path,
        split=split,
        dataset_size=dataset_size
    )
    dataset = dataset.with_transform(
        transform=lambda x: batch_to_tensor(batch=x, columns=columns, pre_transform=pre_transform, post_transform=post_transform),
        columns=list(columns.values()),
    )
    logger.info(f"load {path}[{split}] finished")
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
    )
