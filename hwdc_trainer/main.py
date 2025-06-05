from hwdc_trainer.dataset_loader import load_mnist_dataset


def start_train():
    dataset = load_mnist_dataset()


if __name__ == '__main__':
    start_train()
