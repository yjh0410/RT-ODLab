# PyTorch_YOLO_Tutorial
YOLO 教程

# 简介
这是一个讲解YOLO的入门教程的代码。在这个项目中，我们继承了YOLOv1~v4、YOLOX以及YOLOv7的中心思想，并在此基础上做了适当的修改来实现了结构较为简洁的YOLO检测器。我们希望通过初学者可以通过学习流行的YOLO检测器顺利入门目标检测领域。

**书籍链接**：与本项目代码配套的技术书籍正在被校阅中，请耐心等待。

## 配置运行环境
- 首先，我们建议使用Anaconda来创建一个conda的虚拟环境
```Shell
conda create -n yolo python=3.6
```

- 然后, 请激活已创建的虚拟环境
```Shell
conda activate yolo
```

- 接着，配置环境:
```Shell
pip install -r requirements.txt 
```

项目作者所使用的环境配置:
- PyTorch = 1.9.1
- Torchvision = 0.10.1

为了能够正常运行该项目的代码，请确保您的torch版本为1.x系列。


## 训练配置
|   配置                  |                   |
|------------------------|--------------------|
| 单个GPU上的batch size   | 16                 |
| 初始学习率              | 0.01               |
| Warnmup策略            | 线性warmup策略      |
| 学习率策略              | 线性学习了衰减策略  |
| 训练优化器              | SGD               |
| 多尺度训练              | True              |

## 实验结果
### VOC
- 下载 VOC.
```Shell
cd <PyTorch_YOLO_Tutorial>
cd dataset/scripts/
sh VOC2007.sh
sh VOC2012.sh
```

- 检查 VOC
```Shell
cd <PyTorch_YOLO_Tutorial>
python dataset/voc.py
```

- 使用 VOC 训练模型

For example:
```Shell
python train.py --cuda -d voc --root path/to/VOC -v yolov1 -bs 16 --max_epoch 150 --wp_epoch 1 --eval_epoch 10 --fp16 --ema --multi_scale
```

**P5-Model on COCO:**

| Model  |   Backbone    | Scale |  IP  | Epoch | AP<sup>val<br>0.5 | FPS<sup>3090<br>FP32-bs1 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|--------|---------------|-------|------|-------|-------------------|--------------------------|-------------------|--------------------|--------|
| YOLOv1 | ResNet-18     |  640  |  √   |  150  |       76.7        |                          |   37.8            |   21.3             | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov1_voc.pth) |
| YOLOv2 | DarkNet-19    |  640  |  √   |  150  |       79.8        |                          |   53.9            |   30.9             | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov2_voc.pth) |
| YOLOv3 | DarkNet-53    |  640  |  √   |  150  |       82.0        |                          |   167.4           |   54.9             | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov3_voc.pth) |
| YOLOv4 | CSPDarkNet-53 |  640  |  √   |  150  |       83.6        |                          |   162.7           |   61.5             | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov4_voc.pth) |
| YOLOv5 | CSPDarkNet-L  |  640  |  √   |  150  |       83.8        |                          |   155.6           |   54.2             | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov5_voc.pth) |
| YOLOX  | CSPDarkNet-L  |  640  |  √   |  150  |       84.6        |                          |   155.4           |   54.2             | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolox_voc.pth) |
| YOLOv7 | ELANNet       |  640  |  √   |  150  |       85.5        |                          |   144.6           |   44.0             | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov7_voc.pth) |

*所有的模型都使用了ImageNet预训练权重（IP），所有的FLOPs都是在VOC2007 test数据集上以640x640或1280x1280的输入尺寸来测试的。FPS指标是在一张3090型号的GPU上以batch size=1的输入来测试的，请注意，测速的内容包括模型前向推理、后处理以及NMS操作。*

### COCO
- 下载 COCO.
```Shell
cd <PyTorch_YOLO_Tutorial>
cd dataset/scripts/
sh COCO2017.sh
```

- 检查 COCO
```Shell
cd <PyTorch_YOLO_Tutorial>
python dataset/coco.py
```

当COCO数据集的路径修改正确后，运行上述命令应该会看到COCO数据的可视化图像。

- 使用COCO训练模型

For example:
```Shell
python train.py --cuda -d coco --root path/to/COCO -v yolov1 -bs 16 --max_epoch 150 --wp_epoch 1 --eval_epoch 10 --fp16 --ema --multi_scale
```

**P5-Model on COCO:**

| Model        |   Backbone    | Scale |  IP  | Epoch |  FPS  | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | Weight |
|--------------|---------------|-------|------|-------|-------|------------------------|-------------------|--------|
| YOLOv1       | ResNet-18     |  640  |  √   |  150  |       |        27.9            |       47.5        | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov1_coco.pth) |
| YOLOv2       | DarkNet-19    |  640  |  √   |  150  |       |        32.7            |       50.9        | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov2_coco.pth) |
| YOLOv3       | DarkNet-53    |  640  |  √   |  250  |       |        42.9            |       63.5        | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov3_coco.pth) |
| YOLOv4       | CSPDarkNet-L  |  640  |  √   |  250  |       |        46.6            |       65.8        | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov4_coco.pth) |
| YOLOv5       | CSPDarkNet-53 |  640  |  √   |  250  |       |                        |                   |  |
| YOLOX        | CSPDarkNet-L  |  640  |  √   |  300  |       |        46.6            |       66.1        | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolox_coco.pth) |
| YOLOv7-Nano  | ELANNet-Nano  |  640  |  √   |  300  |       |                        |                   |  |
| YOLOv7-Tiny  | ELANNet-Tiny  |  640  |  √   |  300  |       |                        |                   |  |
| YOLOv7-Large | ELANNet-Large |  640  |  √   |  300  |       |                        |                   |  |
| YOLOv7-Huge  | ELANNet-Huge  |  640  |  √   |  300  |       |                        |                   |  |

*所有的模型都使用了ImageNet预训练权重（IP），所有的FLOPs都是在COCO-val数据集上以640x640或1280x1280的输入尺寸来测试的。FPS指标是在一张3090型号的GPU上以batch size=1的输入来测试的，请注意，测速的内容包括模型前向推理、后处理以及NMS操作。*

## 训练
### 使用单个GPU来训练
```Shell
sh train.sh
```

使用者可以根据自己的情况来调整`train.sh`文件中的配置，以便在自己的本地上顺利训练模型。

如果使用者想查看训练时所使用的数据，可以在训练命令中输入`--vsi_tgt`参数，例如：
```Shell
python train.py --cuda -d coco --root path/to/coco -v yolov1 --vis_tgt
```

### 使用多个GPU来训练
```Shell
sh train_ddp.sh
```

使用者可以根据自己的情况来调整`train_ddp.sh`文件中的配置，以便在自己的本地上顺利训练模型。

**当训练突然中断时**, 使用者可以在训练命令中传入`--resume`参数，并指定最新保存的权重文件（默认为`None`），以便继续训练。例如：

```Shell
python train.py \
        --cuda \
        -d coco \
        -v yolov1 \
        -bs 16 \
        --max_epoch 150 \
        --wp_epoch 1 \
        --eval_epoch 10 \
        --ema \
        --fp16 \
        --resume weights/coco/yolov1/yolov1_epoch_151_39.24.pth
```

## 测试
使用者可以参考下面的给出的例子在相应的数据集上去测试训练好的模型，正常情况下，使用者将会看到检测结果的可视化图像。

```Shell
python test.py -d coco \
               --cuda \
               -v yolov1 \
               --img_size 640 \
               --weight path/to/weight \
               --root path/to/dataset/ \
               --show
```

## 验证
使用者可以参考下面的给出的例子在相应的数据集上去验证训练好的模型，正常情况下，使用者将会看到COCO风格的AP结果输出。

```Shell
python eval.py -d coco-val \
               --cuda \
               -v yolov1 \
               --img_size 640 \
               --weight path/to/weight \
               --root path/to/dataset/ \
               --show
```

如果使用者想测试模型在COCO test-dev数据集上的AP指标，可以遵循以下步骤：

- 将上述命令中的`coco-val`修改为`coco-test`，然后运行；
- 运行结束后，将会得到一个名为`coco_test-dev.json`的文件；
- 将其压缩为一个`.zip`，按照COCO官方的要求修改压缩文件的名称，例如``;
- 按照COCO官方的要求，将该文件上传至官方的服务器去计算AP。

## Demo
本项目在`data/demo/images/`文件夹中提供了一些图片，使用者可以运行下面的命令来测试本地的图片：

```Shell
python demo.py --mode image \
               --path_to_img data/demo/images/ \
               -v yolov1 \
               --img_size 640 \
               --cuda \
               --weight path/to/weight
```

如果使用者想在本地的视频上去做测试，那么你需要将上述命令中的`--mode image`修改为`--mode video`，并给`--path_to_vid`传入视频所在的文件路径，例如：

```Shell
python demo.py --mode video \
               --path_to_img data/demo/videos/your_video \
               -v yolov1 \
               --img_size 640 \
               --cuda \
               --weight path/to/weight
```

如果使用者想用本地的摄像头（如笔记本的摄像头）去做测试，那么你需要将上述命令中的`--mode image`修改为`--mode camera`，例如：

```Shell
python demo.py --mode camera \
               -v yolov1 \
               --img_size 640 \
               --cuda \
               --weight path/to/weight
```
