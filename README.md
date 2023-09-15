# Real-time General Object Detection

English | [简体中文](https://github.com/yjh0410/RT-ODLab/blob/main/README_CN.md)

## Requirements
- We recommend you to use Anaconda to create a conda environment:
```Shell
conda create -n rtcdet python=3.6
```

- Then, activate the environment:
```Shell
conda activate rtcdet
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
cd <RT-ODLab>
cd dataset/scripts/
sh VOC2007.sh
sh VOC2012.sh
```

- Check VOC
```Shell
cd <RT-ODLab>
python dataset/voc.py
```

- Train on VOC

For example:
```Shell
python train.py --cuda -d voc --root path/to/VOCdevkit -m yolov1 -bs 16 --max_epoch 150 --wp_epoch 1 --eval_epoch 10 --fp16 --ema --multi_scale
```

### COCO

- Download COCO.
```Shell
cd <RT-ODLab>
cd dataset/scripts/
sh COCO2017.sh
```

- Check COCO
```Shell
cd <RT-ODLab>
python dataset/coco.py
```

- Train on COCO

For example:
```Shell
python train.py --cuda -d coco --root path/to/COCO -m yolov1 -bs 16 --max_epoch 150 --wp_epoch 1 --eval_epoch 10 --fp16 --ema --multi_scale
```

## Train
### Single GPU
```Shell
sh train_single_gpu.sh
```

You can change the configurations of `train_single_gpu.sh`, according to your own situation.

You also can add `--vis_tgt`  to check the images and targets during the training stage. For example:
```Shell
python train.py --cuda -d coco --root path/to/coco -m yolov1 --vis_tgt
```

### Multi GPUs
```Shell
sh train_multi_gpus.sh
```

You can change the configurations of `train_multi_gpus.sh`, according to your own situation.

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
