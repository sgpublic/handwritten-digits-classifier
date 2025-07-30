# Vision Models

这是一个图像相关的任务模型源码，用于归档学习过程，包含：

+ [MNIST](https://huggingface.co/datasets/ylecun/mnist)
+ [CIFAR-10](https://huggingface.co/datasets/uoft-cs/cifar10)

使用模型：

+ ResNet
+ VGG

## 开发环境

+ Debian 12
+ Python 3.10
+ cuda-toolkit 12.8、cudnn 9（RTX 2060）

## 食用方法

安装依赖：

```shell
pixi install
```

默认环境将安装 GPU 计算相关依赖，若要使用 CPU 计算，请执行：

```shell
pixi install --environment cpu
```

开始训练：

```shell
CORE_DEVICE=cuda pixi run train_mnist_resnet # 使用 GPU 训练
CORE_DEVICE=cpu pixi run train_mnist_resnet --environment cpu # 使用 CPU 训练
```

启动 WebUI：

```shell
CORE_DEVICE=cuda pixi run webui_mnist_resnet # 使用 GPU 推理
CORE_DEVICE=cpu pixi run webui_mnist_resnet --environment cpu # 使用 CPU 推理
```

环境变量：

| 变量名称              | 变量解释                                                  | 默认值             |
|-------------------|-------------------------------------------------------|-----------------|
| CORE_DEBUG        | 是否打印 DEBUG 日志                                         | `false`         |
| CORE_DEVICE       | 训练/推理使用的设备类型，可选 `cuda`、`cpu` 等，当 cuda 不可用时自动退回 CPU 计算 | `cuda`          |
| CORE_DATASET_TYPE | 任务类型，可选 `MNIST`、`CIFAR_10`                            | `MNIST`         |
| CORE_MODEL_TYPE   | 模型类型，可选 `ResNet_Custom`、`VGG_Custom`                  | `ResNet_Custom` |

> PS：当使用环境变量 `CORE_DATASET_TYPE` 和 `CORE_MODEL_TYPE` 时，请使用 `pixi run webui` 或 `pixi run train` 启动任务。

训练相关环境变量：

| 变量名称                              | 变量解释                | 默认值   |
|-----------------------------------|---------------------|-------|
| TRAINER_LEARN_RATE                | 学习率                 | 1e-2  |
| TRAINER_DATASET_BATCH_SIZE        | 训练时每批次数据集大小         | 32    |
| TRAINER_DATASET_TEST_DATASET_SIZE | 训练时使用的测试集数量         | 1000  |
| TRAINER_DATASET_MAX_EPOCHS        | 训练时最大批次数量           | 50    |
| TRAINER_MODEL_ACCURACY_THRESHOLD  | 当测试集准确率达到此设定数值时训练停止 | 0.995 |

WebUI 相关环境变量：

| 变量名称                        | 变量解释            | 默认值       |
|-----------------------------|-----------------|-----------|
| GRADIO_LISTEN_HOST          | WebUI 监听地址      | `0.0.0.0` |
| GRADIO_LISTEN_PORT          | WebUI 监听端口      | 7890      |
| GRADIO_MODEL_USE_PRETRAINED | WebUI 是否使用预训练模型 | `true`    |

手写数字模型相关环境变量：

| 变量名称                               | 变量解释                                                                                             | 默认值  |
|------------------------------------|--------------------------------------------------------------------------------------------------|------|
| MNIST_DATASET_RANDOM_ROTATE        | 对训练集随机进行旋转变换，将在 `[-MNIST_DATASET_RANDOM_ROTATE, +MNIST_DATASET_RANDOM_ROTATE]` 范围内随机变换           | 10.0 |
| MNIST_DATASET_RANDOM_SCALE         | 对训练集随机进行缩放变换，将在 `[1.0 - MNIST_DATASET_RANDOM_SCALE, 1.0 + MNIST_DATASET_RANDOM_SCALE]` 倍率范围内随机变换 | 0.2  |
| MNIST_DATASET_RANDOM_ELASTIC_ALPHA | 对训练集随机进行弹性变换的强度                                                                                  | 34.0 |
| MNIST_DATASET_RANDOM_ELASTIC_SIGMA | 对训练集随机进行弹性变换的平滑度                                                                                 | 4.0  |
