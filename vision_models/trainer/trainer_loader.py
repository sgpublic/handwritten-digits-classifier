from vision_models.core.config import CORE_DATASET_TYPE
from vision_models.core.model_trainer import ModelTrainer
from vision_models.core.types.dataset_type import DatasetType
from vision_models.models.mnist.mnist_model_trainer import MnistModelTrainer


def load_trainer(dataset_type: DatasetType = CORE_DATASET_TYPE) -> ModelTrainer:
    trainer: ModelTrainer
    match dataset_type:
        case DatasetType.MNIST:
            trainer = MnistModelTrainer()
        case DatasetType.CIFAR_10:
            trainer = MnistModelTrainer()
    return trainer
