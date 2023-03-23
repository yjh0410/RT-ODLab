# PyTorch_YOLO_Tutorial
YOLO Tutorial

English | [简体中文](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/blob/main/README_CN.md)

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

| Model  | Scale |  IP  | AP<sup>val<br>0.5:0.95 | AP<sup>test<br>0.5:0.95 | FPS<sup>3090<br>FP32-bs1 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|--------|-------|------|------------------------|-------------------------|--------------------------|-------------------|--------------------|--------|
| YOLOv1 |  640  |  √   |   35.5                 |                         |     100                  |   9.0             |   2.3              |  |
| YOLOv2 |  640  |  √   |                        |                         |                          |   33.5            |   8.3              |  |
| YOLOv3 |  640  |  √   |                        |                         |                          |   86.7            |   23.0             |  |
| YOLOv4 |  640  |  √   |                        |                         |                          |   175.4           |   46.5             |  |

*All models are trained with ImageNet pretrained weight (IP). All FLOPs are measured with a 640x640 image size on COCO val2017. The FPS is measured with batch size 1 on 3090 GPU from the model inference to the NMS operation.*



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

| Model  | Scale |  IP  | mAP | FPS<sup>3090<br>FP32-bs1 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|--------|-------|------|-----|--------------------------|-------------------|--------------------|--------|
| YOLOv1 |  640  |  √   |     |                          |   37.8            |   21.3             |  |
| YOLOv2 |  640  |  √   |     |                          |   53.9            |   30.9             |  |
| YOLOv3 |  640  |  √   |     |                          |                   |                    |  |
| YOLOv4 |  640  |  √   |     |                          |                   |                    |  |

*All models are trained with ImageNet pretrained weight (IP). All FLOPs are measured with a 640x640 image size on VOC2007 test. The FPS is measured with batch size 1 on 3090 GPU from the model inference to the NMS operation.*


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
