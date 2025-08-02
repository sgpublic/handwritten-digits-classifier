import os

CIFAR10_RESNET_WEIGHT_DECAY: float = float(os.getenv("CIFAR10_RESNET_WEIGHT_DECAY", "1e-4"))
CIFAR10_RESNET_DROPOUT: float = float(os.getenv("CIFAR10_RESNET_DROPOUT", "0.3"))
