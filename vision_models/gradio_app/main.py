from vision_models.core.config import CORE_DATASET_TYPE
from vision_models.core.types.dataset_type import DatasetType
from vision_models.gradio_app.gradio_app import GradioApp
from vision_models.models.cifar10.cifar10_gradio_app import Cifar10GradioApp
from vision_models.models.mnist.mnist_gradio_app import MnistGradioApp

def main():
    app: GradioApp
    match CORE_DATASET_TYPE:
        case DatasetType.MNIST:
            app = MnistGradioApp()
        case DatasetType.CIFAR_10:
            app = Cifar10GradioApp()
    app.launch()

if __name__ == '__main__':
    main()
