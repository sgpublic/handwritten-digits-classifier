from enum import Enum


class DatasetType(Enum):
    MNIST = "mnist"
    CIFAR_10 = "cifar-10"

class DatasetColumnType(Enum):
    IMAGE = "image"
    LABEL = "label"

class DatasetSplitType(Enum):
    TRAIN = "train"
    TEST = "test"
