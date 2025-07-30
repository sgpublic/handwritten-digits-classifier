from vision_models.core.config import CORE_DATASET_TYPE
from vision_models.core.types.dataset_type import DatasetType
from vision_models.gradio_app.gradio_app import GradioApp
from vision_models.models.mnist.mnist_gradio_app import MnistGradioApp

if __name__ == '__main__':
    app: GradioApp
    match CORE_DATASET_TYPE:
        case DatasetType.MNIST:
            app = MnistGradioApp()
        case DatasetType.CIFAR_10:
            app = MnistGradioApp()
    app.launch()
