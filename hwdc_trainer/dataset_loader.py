from datasets import DownloadConfig, load_dataset
from torch.utils.data import DataLoader

from hwdc.core.logger import create_logger

logger = create_logger(__name__)

_dl_config = DownloadConfig(
    resume_download=True,
    force_download=False,
)


# https://huggingface.co/datasets/ylecun/mnist
# https://huggingface.co/docs/datasets/use_with_pytorch#stream-data
def load_mnist_dataset(
        split: str = 'train',
        batch_size: int = 32,
):
    logger.info("loading mnist dataset...")
    dataset = load_dataset(
        path='mnist',
        download_config=_dl_config,
        streaming=True,
    )
    logger.info("load mnist dataset finished")
    return DataLoader(dataset=dataset[split], batch_size=batch_size)
