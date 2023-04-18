# PyTorch_YOLO_Tutorial
YOLO Tutorial

English | [简体中文](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/blob/main/README_CN.md)

# Introduction
Here is the source code for an introduction to YOLO. We adopted the core concepts of **YOLOv1~v4**, **YOLOX** and **YOLOv7** for this project and made the necessary adjustments. By learning how to construct the well-known YOLO detector, we hope that newcomers can enter the field of object detection without any difficulty.

**Book**: The technical books that go along with this project's code is being reviewed, please be patient.

## Requirements
- We recommend you to use Anaconda to create a conda environment:
```Shell
conda create -n yolo python=3.6
```

- Then, activate the environment:
```Shell
conda activate yolo
```

- Requirements:
```Shell
pip install -r requirements.txt 
```

My environment:
- PyTorch = 1.9.1
- Torchvision = 0.10.1

At least, please make sure your torch is version 1.x.

## Training Configuration
|   Configuration         |                            |
|-------------------------|----------------------------|
| Per GPU Batch Size      | 16                         |
| Init Lr                 | 0.01                       |
| Warmup Scheduler        | Linear                     |
| Lr Scheduler            | Linear                     |
| Optimizer               | SGD                        |
| Multi Scale Train       | True                       |


## Experiments
### VOC
- Download VOC.
```Shell
cd <PyTorch_YOLO_Tutorial>
cd dataset/scripts/
sh VOC2007.sh
sh VOC2012.sh
```

- Check VOC
```Shell
cd <PyTorch_YOLO_Tutorial>
python dataset/voc.py
```

- Train on VOC

For example:
```Shell
python train.py --cuda -d voc --root path/to/VOCdevkit -v yolov1 -bs 16 --max_epoch 150 --wp_epoch 1 --eval_epoch 10 --fp16 --ema --multi_scale
```

| Model  |   Backbone    | Scale |  IP  | Epoch | AP<sup>val<br>0.5 | FPS<sup>3090<br>FP32-bs1 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|--------|---------------|-------|------|-------|-------------------|--------------------------|-------------------|--------------------|--------|
| YOLOv1 | ResNet-18     |  640  |  √   |  150  |       76.7        |                          |   37.8            |   21.3             | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov1_voc.pth) |
| YOLOv2 | DarkNet-19    |  640  |  √   |  150  |       79.8        |                          |   53.9            |   30.9             | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov2_voc.pth) |
| YOLOv3 | DarkNet-53    |  640  |  √   |  150  |       82.0        |                          |   167.4           |   54.9             | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov3_voc.pth) |
| YOLOv4 | CSPDarkNet-53 |  640  |  √   |  150  |       83.6        |                          |   162.7           |   61.5             | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov4_voc.pth) |
| YOLOv5 | CSPDarkNet-L  |  640  |  √   |  150  |       83.8        |                          |   155.6           |   54.2             | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov5_voc.pth) |
| YOLOX  | CSPDarkNet-L  |  640  |  √   |  150  |       84.6        |                          |   155.4           |   54.2             | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolox_voc.pth) |
| YOLOv7 | ELANNet       |  640  |  √   |  150  |       85.5        |                          |   144.6           |   44.0             | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov7_voc.pth) |

*All models are trained with ImageNet pretrained weight (IP). All FLOPs are measured with a 640x640 image size on VOC2007 test. The FPS is measured with batch size 1 on 3090 GPU from the model inference to the NMS operation.*


### COCO
- Download COCO.
```Shell
cd <PyTorch_YOLO_Tutorial>
cd dataset/scripts/
sh COCO2017.sh
```

- Check COCO
```Shell
cd <PyTorch_YOLO_Tutorial>
python dataset/coco.py
```

- Train on COCO

For example:
```Shell
python train.py --cuda -d coco --root path/to/COCO -v yolov1 -bs 16 --max_epoch 150 --wp_epoch 1 --eval_epoch 10 --fp16 --ema --multi_scale
```

| Model        |   Backbone    | Scale |  IP  | Epoch |  FPS  | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | Weight |
|--------------|---------------|-------|------|-------|-------|------------------------|-------------------|--------|
| YOLOv1       | ResNet-18     |  640  |  √   |  150  |       |        27.9            |       47.5        | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov1_coco.pth) |
| YOLOv2       | DarkNet-19    |  640  |  √   |  150  |       |        32.7            |       50.9        | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov2_coco.pth) |
| YOLOv3       | DarkNet-53    |  640  |  √   |  250  |       |        42.9            |       63.5        | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov3_coco.pth) |
| YOLOv4       | CSPDarkNet-53 |  640  |  √   |  250  |       |                        |                   |  |
| YOLOv5       | CSPDarkNet-L  |  640  |  √   |  250  |       |        46.6            |       65.8        | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov4_coco.pth) |
| YOLOX        | CSPDarkNet-L  |  640  |  √   |  300  |       |        46.6            |       66.1        | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolox_coco.pth) |
| YOLOv7-Nano  | ELANNet-Nano  |  640  |  √   |  300  |       |                        |                   |  |
| YOLOv7-Tiny  | ELANNet-Tiny  |  640  |  √   |  300  |       |                        |                   |  |
| YOLOv7-Large | ELANNet-Large |  640  |  √   |  300  |       |                        |                   |  |
| YOLOv7-Huge  | ELANNet-Huge  |  640  |  √   |  300  |       |                        |                   |  |

*All models are trained with ImageNet pretrained weight (IP). All FLOPs are measured with a 640x640 image size on COCO val2017. The FPS is measured with batch size 1 on 3090 GPU from the model inference to the NMS operation.*


## Train
### Single GPU
```Shell
sh train.sh
```

You can change the configurations of `train.sh`, according to your own situation.

You also can add `--vis_tgt`  to check the images and targets during the training stage. For example:
```Shell
python train.py --cuda -d coco --root path/to/coco -v yolov1 --vis_tgt
```

### Multi GPUs
```Shell
sh train_ddp.sh
```

You can change the configurations of `train_ddp.sh`, according to your own situation.

**In the event of a training interruption**, you can pass `--resume` the latest training
weight path (`None` by default) to resume training. For example:

```Shell
python train.py \
        --cuda \
        -d coco \
        -v yolov1 \
        -bs 16 \
        --max_epoch 300 \
        --wp_epoch 3 \
        --eval_epoch 10 \
        --ema \
        --fp16 \
        --resume weights/coco/yolov1/yolov1_epoch_151_39.24.pth
```

Then, training will continue from 151 epoch.

## Test
```Shell
python test.py -d coco \
               --cuda \
               -v yolov1 \
               --img_size 640 \
               --weight path/to/weight \
               --root path/to/dataset/ \
               --show
```

## Evaluation
```Shell
python eval.py -d coco-val \
               --cuda \
               -v yolov1 \
               --img_size 640 \
               --weight path/to/weight \
               --root path/to/dataset/ \
               --show
```

## Demo
I have provide some images in `data/demo/images/`, so you can run following command to run a demo:

```Shell
python demo.py --mode image \
               --path_to_img data/demo/images/ \
               -v yolov1 \
               --img_size 640 \
               --cuda \
               --weight path/to/weight
```

If you want run a demo of streaming video detection, you need to set `--mode` to `video`, and give the path to video `--path_to_vid`。

```Shell
python demo.py --mode video \
               --path_to_img data/demo/videos/your_video \
               -v yolov1 \
               --img_size 640 \
               --cuda \
               --weight path/to/weight
```

If you want run video detection with your camera, you need to set `--mode` to `camera`。

```Shell
python demo.py --mode camera \
               -v yolov1 \
               --img_size 640 \
               --cuda \
               --weight path/to/weight
```
