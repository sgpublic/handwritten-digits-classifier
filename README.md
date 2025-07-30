# Hand written digits classifier

这是一个手写数字分类的图像分类任务模型。

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
HWDC_DEVICE=gpu pixi run train # 使用 GPU 训练
HWDC_DEVICE=cpu pixi run train --environment cpu # 使用 CPU 训练
```

启动 WebUI：

```shell
HWDC_DEVICE=gpu pixi run webui # 使用 GPU 推理
HWDC_DEVICE=cpu pixi run webui --environment cpu # 使用 CPU 推理
```

环境变量：

| 变量名称                              | 变量解释                                                                                            | 默认值       |
|-----------------------------------|-------------------------------------------------------------------------------------------------|-----------|
| HWDC_DEBUG                        | 是否打印 DEBUG 日志                                                                                   | `false`   |
| HWDC_DEVICE                       | 训练/推理使用的设备类型，可选 `cuda`、`cpu` 等，当 GPU 不可用时自动退回 CPU 计算                                            | `gpu`     |
| HWDC_DATASET_BATCH_SIZE           | 训练时每批次数据集大小                                                                                     | 32        |
| HWDC_DATASET_TEST_DATASET_SIZE    | 训练时使用的测试集数量                                                                                     | 10000     |
| HWDC_DATASET_MAX_EPOCHS           | 训练时最大批次数量                                                                                       | 50        |
| HWDC_DATASET_RANDOM_ROTATE        | 训练时对训练集随机进行旋转变换，将在 `[-HWDC_DATASET_RANDOM_ROTATE, +HWDC_DATASET_RANDOM_ROTATE]` 范围内随即变换         | 10.0      |
| HWDC_DATASET_RANDOM_SCALE         | 训练时对训练集随机进行缩放变换，将在 `[1.0 - HWDC_DATASET_RANDOM_SCALE, 1.0 + HWDC_DATASET_RANDOM_SCALE]` 范围内随即变换 | 0.2       |
| HWDC_DATASET_RANDOM_ELASTIC_ALPHA | 训练时对训练集随机进行弹性变换的强度                                                                              | 34.0      |
| HWDC_DATASET_RANDOM_ELASTIC_SIGMA | 训练时对训练集随机进行弹性变换的平滑度                                                                             | 4.0       |
| HWDC_MODEL_ACCURACY_THRESHOLD     | 当测试集准确率达到此设定数值时训练停止                                                                             | 0.99      |
| HWDC_MODEL_SAVE_INTERVAL          | 训练时模型保存步长（批次）                                                                                   | 1         |
| HWDC_GRADIO_HOST                  | WebUI 监听地址                                                                                      | `0.0.0.0` |
| HWDC_MODEL_USE_PRETRAINED         | WebUI 是否使用预训练模型                                                                                 | `true`    |
