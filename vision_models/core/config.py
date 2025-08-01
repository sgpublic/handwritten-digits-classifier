import os
from distutils.util import strtobool

from vision_models.core.types.dataset_type import DatasetType
from vision_models.core.types.model_type import ModelType

CORE_DEBUG: bool = bool(strtobool(os.getenv("MNIST_DEBUG", "false")))
CORE_DEVICE: str = os.getenv("MNIST_DEVICE", "cuda")
CORE_DATASET_TYPE: DatasetType = DatasetType[os.getenv("CORE_DATASET_TYPE")]
CORE_MODEL_TYPE: ModelType = ModelType[os.getenv("CORE_MODEL_TYPE")]
