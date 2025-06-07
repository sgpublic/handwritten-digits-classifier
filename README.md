# Hand written digits classifier

这是一个手写数字分类的图像分类任务模型。

## 开发环境

+ Debian 12
+ Python 3.10
+ cuda-toolkit 12.8、cudnn 9（RTX 2060）

## 食用方法

安装依赖：

```shell
poetry install # 安装基础依赖
poetry run poe pytorch-cpu-alimirror # 安装 PyTorch，使用 CPU 计算（可选：pytorch-cpu、pytorch-cu128、pytorch-cpu-alimirror、pytorch-cu128-alimirror）
```

开始训练：

```shell
poetry run train
```

启动 WebUI：

```shell
poetry run webui
```

环境变量：

| 变量名称                      | 变量解释                           | 默认值     |
|---------------------------|--------------------------------|---------|
| HWDC_DEBUG                | 是否打印 DEBUG 日志                  | `false` |
| HWDC_DEVICE               | 训练/推理使用的设备类型，可选 `cuda`、`cpu` 等 | `cpu`   |
| HWDC_DATASET_BATCH_SIZE   | 训练时每批次数据集大小                    | 100     |
| HWDC_DATASET_EPOCHS       | 训练时批次数量                        | 10      |
| HWDC_MODEL_SAVE_INTERVAL  | 训练时模型保存步长（批次）                  | 1       |
| HWDC_MODEL_USE_PRETRAINED | WebUI 是否使用预训练模型                | `true`  |
