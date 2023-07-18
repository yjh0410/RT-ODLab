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
python train.py --cuda -d voc --root path/to/VOCdevkit -m yolov1 -bs 16 --max_epoch 150 --wp_epoch 1 --eval_epoch 10 --fp16 --ema --multi_scale
```

| Model        |   Backbone          | Scale |  IP  | Epoch | AP<sup>val<br>0.5 | FPS<sup>3090<br>FP32-bs1 | Weight |
|--------------|---------------------|-------|------|-------|-------------------|--------------------------|--------|
| YOLOv1       | ResNet-18           |  640  |  √   |  150  |       76.7        |                          | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov1_voc.pth) |
| YOLOv2       | DarkNet-19          |  640  |  √   |  150  |       79.8        |                          | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov2_voc.pth) |
| YOLOv3       | DarkNet-53          |  640  |  √   |  150  |       82.0        |                          | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov3_voc.pth) |
| YOLOv4       | CSPDarkNet-53       |  640  |  √   |  150  |       83.6        |                          | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov4_voc.pth) |
| YOLOX-L      | CSPDarkNet-L        |  640  |  √   |  150  |       84.6        |                          | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolox_voc.pth) |
| YOLOv7-Large | ELANNet-Large       |  640  |  √   |  150  |       86.0        |                          | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov7_large_voc.pth) |

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
python train.py --cuda -d coco --root path/to/COCO -m yolov1 -bs 16 --max_epoch 150 --wp_epoch 1 --eval_epoch 10 --fp16 --ema --multi_scale
```

#### YOLOv3:

| Model       |  Backbone    | Batch | Scale | Epoch | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|-------------|--------------|-------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| YOLOv3-Tiny | DarkNet-Tiny | 1xb16 |  640  |  250  |        25.4            |       43.4        |   7.0             |   2.3              | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov3_t_coco.pth) |
| YOLOv3      | DarkNet-53   | 1xb16 |  640  |  250  |        42.9            |       63.5        |   167.4           |   54.9             | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov3_coco.pth) |

#### YOLOv4:

| Model       |  Backbone       | Batch | Scale | Epoch | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|-------------|-----------------|-------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| YOLOv4-Tiny | CSPDarkNet-Tiny | 1xb16 |  640  |  250  |        31.0            |       49.1        |   8.1             |   2.9              | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov4_t_coco.pth) |
| YOLOv4      | CSPDarkNet-53   | 1xb16 |  640  |  250  |        46.6            |       65.8        |   162.7           |   61.5             | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov4_coco.pth) |

#### YOLOv5:

| Model     |  Backbone    | Batch | Scale | Epoch | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|-----------|--------------|-------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| YOLOv5-N  | CSPDarkNet-N | 1xb16 |  640  |  250  |         29.8           |       47.1        |   7.7             |   2.4              | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov5_n_coco.pth) |
| YOLOv5-S  | CSPDarkNet-S | 1xb16 |  640  |  250  |         37.8           |       56.5        |   27.1            |   9.0              | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov5_s_coco.pth) |
| YOLOv5-M  | CSPDarkNet-M | 1xb16 |  640  |  250  |         43.5           |       62.5        |   74.3            |   25.4             | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov5_m_coco.pth) |
| YOLOv5-L  | CSPDarkNet-L | 1xb16 |  640  |  250  |         46.7           |       65.5        |   155.6           |   54.2             | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov5_l_coco.pth) |

*For **YOLOv5-M** and **YOLOv5-L**, increasing the batch size may improve performance. Due to my computing resources, I can only set the batch size to 16.*

#### YOLOX:

| Model   |   Backbone    | Batch | Scale | Epoch | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|---------|---------------|-------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| YOLOX-N | CSPDarkNet-N  | 8xb8  |  640  |  300  |         30.4           |       48.9        |   7.5             |   2.3              | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolox_n_coco.pth) |
| YOLOX-S | CSPDarkNet-S  | 8xb8  |  640  |  300  |         39.0           |       58.8        |   26.8            |   8.9              | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolox_s_coco.pth) |
| YOLOX-M | CSPDarkNet-M  | 1xb16 |  640  |  300  |         44.6           |       63.8        |   74.3            |   25.4             | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolox_m_coco.pth) |
| YOLOX-L | CSPDarkNet-L  | 1xb16 |  640  |  300  |         46.9           |       65.9        |   155.4           |   54.2             | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolox_l_coco.pth) |

*For **YOLOX-M** and **YOLOX-L**, increasing the batch size may improve performance. Due to my computing resources, I can only set the batch size to 16.*

#### YOLOv7:

| Model       | Backbone      | Batch | Scale | Epoch | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|-------------|---------------|-------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| YOLOv7-Tiny | ELANNet-Tiny  | 1xb16 |  640  |  300  |         38.0           |       56.8        |   22.6            |   7.9              | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov7_tiny_coco.pth) |
| YOLOv7      | ELANNet-Large | 1xb16 |  640  |  300  |         48.0           |       67.5        |   144.6           |   44.0             | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov7_large_coco.pth) |

* While YOLOv7 incorporates several technical details, such as anchor box, SimOTA, AuxiliaryHead, and RepConv, I found it too challenging to fully reproduce. Instead, I created a simpler version of YOLOv7 using an anchor-free structure and SimOTA. As a result, my reproduction had poor performance due to the absence of the other technical details. However, since it was only intended as a tutorial, I am not too concerned about this gap.*

#### My YOLO:

| Model    | Scale | Batch | AP<sup>test<br>0.5:0.95 | AP<sup>test<br>0.5 | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|----------|-------|-------|-------------------------|--------------------|------------------------|-------------------|-------------------|--------------------|--------|
| YOLOvx-N |  640  | 8xb16 |                         |                    |                        |                   |      9.1          |        2.4         |  |
| YOLOvx-T |  640  | 8xb16 |                         |                    |                        |                   |      18.9         |        5.1         |  |
| YOLOvx-S |  640  | 8xb16 |                         |                    |                        |                   |      33.6         |        9.0         |  |
| YOLOvx-M |  640  | 8xb16 |         48.3            |        67.0        |          48.1          |        66.9       |      87.4         |        23.6        | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolovx_m_coco.pth) |
| YOLOvx-L |  640  | 8xb16 |         50.2            |        68.6        |          50.0          |        68.4       |      176.6        |        47.6        | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolovx_l_coco.pth) |
| YOLOvx-X |  640  |       |                         |                    |                        |                   |                   |                    |  |

- We use `AdamW` optimizer with `per_image_lr=0.001 / 64` and `linear` learning rate decay scheduler to train all models with 300 epoch.
- We use `YOLOv5-style Mosaic augmentation` and `YOLOX-style Mixup augmentation` wihout rotation.
- Due to my limited computing resources, I can not train `YOLOvx-X` with the setting of `batch size=128`.

#### Redesigned RT-DETR:

| Model     | Scale | Batch | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|-----------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| RT-DETR-N |  640  |       |                        |                   |                   |                    |  |
| RT-DETR-S |  640  |       |                        |                   |                   |                    |  |
| RT-DETR-M |  640  |       |                        |                   |                   |                    |  |
| RT-DETR-L |  640  |       |                        |                   |                   |                    |  |

#### Necessary instructions：

- *All models are trained with ImageNet pretrained weight (IP). All FLOPs are measured with a 640x640 image size on COCO val2017. The FPS is measured with batch size 1 on 3090 GPU from the model inference to the NMS operation.*

- *The reproduced YOLOv5's head is the **Decoupled Head**, which is why the FLOPs and Params are higher than the official YOLOv5. Due to my limited computing resources, I can not align the training configuration with the official YOLOv5, so I cannot fully replicate the official performance. The YOLOv5 I reproduce is for learning purposes only.*

- *Due to my limited computing resources, I had to abandon training on other YOLO detectors, including YOLOv7-Huge and YOLOv8-Nano~Large. If you are interested in these models and have trained them using the code from this project, I would greatly appreciate it if you could share the trained weight files with me.*

- *Using a larger batch size may improve the performance of large models, such as YOLOv5-L, YOLOX-L and YOLOv7-L. Due to my computing resources, I can only set the batch size to 16.*

## Train
### Single GPU
```Shell
sh train.sh
```

You can change the configurations of `train.sh`, according to your own situation.

You also can add `--vis_tgt`  to check the images and targets during the training stage. For example:
```Shell
python train.py --cuda -d coco --root path/to/coco -m yolov1 --vis_tgt
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
        -m yolov1 \
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
               -m yolov1 \
               --img_size 640 \
               --weight path/to/weight \
               --root path/to/dataset/ \
               --show
```

For YOLOv7, since it uses the RepConv in PaFPN, you can add `--fuse_repconv` to fuse the RepConv block.
```Shell
python test.py -d coco \
               --cuda \
               -m yolov7_large \
               --fuse_repconv \
               --img_size 640 \
               --weight path/to/weight \
               --root path/to/dataset/ \
               --show
```


## Evaluation
```Shell
python eval.py -d coco-val \
               --cuda \
               -m yolov1 \
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
               --cuda \
               --img_size 640 \
               -m yolov2 \
               --weight path/to/weight \
               --show
```

If you want run a demo of streaming video detection, you need to set `--mode` to `video`, and give the path to video `--path_to_vid`。

```Shell
python demo.py --mode video \
               --path_to_vid data/demo/videos/your_video \
               --cuda \
               --img_size 640 \
               -m yolov2 \
               --weight path/to/weight \
               --show \
               --gif
```

If you want run video detection with your camera, you need to set `--mode` to `camera`。

```Shell
python demo.py --mode camera \
               --cuda \
               --img_size 640 \
               -m yolov2 \
               --weight path/to/weight \
               --show \
               --gif
```

### Detection visualization
* Detector: YOLOv2

Command：

```Shell
python demo.py --mode video \
                --path_to_vid ./dataset/demo/videos/000006.mp4 \
               --cuda \
               --img_size 640 \
               -m yolov2 \
               --weight path/to/weight \
               --show \
               --gif
```

Results:

![image](./img_files/video_detection_demo.gif)

## Tracking
Our project also supports **multi-object tracking** tasks. We use the YOLO of this project as the detector, following the "tracking-by-detection" framework, and use the simple and efficient **ByteTrack** as the tracker.

* images tracking
```Shell
python track.py --mode image \
                --path_to_img path/to/images/ \
                --cuda \
                -size 640 \
                -dt yolov2 \
                -tk byte_tracker \
                --weight path/to/coco_pretrained/ \
                --show \
                --gif
```

* video tracking

```Shell
python track.py --mode video \
                --path_to_img path/to/video/ \
                --cuda \
                -size 640 \
                -dt yolov2 \
                -tk byte_tracker \
                --weight path/to/coco_pretrained/ \
                --show \
                --gif
```

* camera tracking

```Shell
python track.py --mode camera \
                --cuda \
                -size 640 \
                -dt yolov2 \
                -tk byte_tracker \
                --weight path/to/coco_pretrained/ \
                --show \
                --gif
```

### Tracking visualization
* Detector: YOLOv2
* Tracker: ByteTracker
* Device: i5-12500H CPU

Command：

```Shell
python track.py --mode video \
                --path_to_img ./dataset/demo/videos/000006.mp4 \
                -size 640 \
                -dt yolov2 \
                -tk byte_tracker \
                --weight path/to/coco_pretrained/ \
                --show \
                --gif
```

Results:

![image](./img_files/video_tracking_demo.gif)


## Train on custom dataset
Besides the popular datasets, we can also train the model on ourself dataset. To achieve this goal, you should follow these steps:
- Step-1: Prepare the images (JPG/JPEG/PNG ...) and use `labelimg` to make XML format annotation files.

```
OurDataset
|_ train
|  |_ images     
|     |_ 0.jpg
|     |_ 1.jpg
|     |_ ...
|  |_ annotations
|     |_ 0.xml
|     |_ 1.xml
|     |_ ...
|_ val
|  |_ images     
|     |_ 0.jpg
|     |_ 1.jpg
|     |_ ...
|  |_ annotations
|     |_ 0.xml
|     |_ 1.xml
|     |_ ...
|  ...
```

- Step-2: Convert ourdataset to COCO format.

```Shell
cd <PyTorch_YOLO_Tutorial_HOME>
cd tools
# convert train split
python convert_ours_to_coco.py --root path/to/dataset/ --split train
# convert val split
python convert_ours_to_coco.py --root path/to/dataset/ --split val
```
Then, we can get a `train.json` file and a `val.json` file, as shown below.
```
OurDataset
|_ train
|  |_ images     
|     |_ 0.jpg
|     |_ 1.jpg
|     |_ ...
|  |_ annotations
|     |_ 0.xml
|     |_ 1.xml
|     |_ ...
|     |_ train.json
|_ val
|  |_ images     
|     |_ 0.jpg
|     |_ 1.jpg
|     |_ ...
|  |_ annotations
|     |_ 0.xml
|     |_ 1.xml
|     |_ ...
|     |_ val.json
|  ...
```

- Step-3 Define our class labels.

Please open `dataset/ourdataset.py` file and change `our_class_labels = ('cat',)` according to our definition of categories.

- Step-4 Check

```Shell
cd <PyTorch_YOLO_Tutorial_HOME>
cd dataset
# convert train split
python ourdataset.py --root path/to/dataset/ --split train
# convert val split
python ourdataset.py --root path/to/dataset/ --split val
```

- Step-5 **Train**

For example:

```Shell
cd <PyTorch_YOLO_Tutorial_HOME>
python train.py --root path/to/dataset/ -d ourdataset -m yolov1 -bs 16 --max_epoch 100 --wp_epoch 1 --eval_epoch 5 -p path/to/yolov1_coco.pth
```

- Step-6 **Test**

For example:

```Shell
cd <PyTorch_YOLO_Tutorial_HOME>
python test.py --root path/to/dataset/ -d ourdataset -m yolov1 --weight path/to/checkpoint --show
```

- Step-7 **Eval**

For example:

```Shell
cd <PyTorch_YOLO_Tutorial_HOME>
python eval.py --root path/to/dataset/ -d ourdataset -m yolov1 --weight path/to/checkpoint
```


## Deployment
1. [ONNX export and an ONNXRuntime](./deployment/ONNXRuntime/)
