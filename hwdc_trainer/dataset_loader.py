from datasets import DownloadConfig, load_dataset
from torch.utils.data import DataLoader

from hwdc.core.logger import create_logger
from hwdc.core.tensor import batch_to_tensor

logger = create_logger(__name__)

_dl_config = DownloadConfig(
    resume_download=True,
    force_download=False,
)


# https://huggingface.co/datasets/ylecun/mnist
# https://huggingface.co/docs/datasets/use_with_pytorch#stream-data
def load_mnist_dataset(
        split: str = "train",
        batch_size: int = 100,
) -> DataLoader:
    logger.info("loading mnist dataset...")
    dataset = load_dataset(
        path="mnist",
        download_config=_dl_config,
    )
    dataset = dataset[split]
    dataset = dataset.with_transform(
        transform=batch_to_tensor,
        columns=["image", "label"],
    )
    logger.info("load mnist dataset finished")
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
    )
