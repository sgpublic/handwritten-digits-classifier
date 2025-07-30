from distutils.util import strtobool
from os import environ

from vision_models.core.types.dataset_type import DatasetType
from vision_models.core.types.model_type import ModelType

CORE_DEBUG: bool = bool(strtobool(environ.get("MNIST_DEBUG", "false")))
CORE_DEVICE: str = environ.get("MNIST_DEVICE", "cuda")
CORE_DATASET_TYPE: DatasetType = DatasetType[environ.get("CORE_DATASET_TYPE", DatasetType.MNIST.name)]
CORE_MODEL_TYPE: ModelType = ModelType[environ.get("CORE_MODEL_TYPE", ModelType.ResNet_Custom.name)]
