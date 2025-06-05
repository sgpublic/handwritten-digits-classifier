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
poetry run webui_local # 使用本地训练的模型
poetry run webui_pretrained # 使预训练模型
```
