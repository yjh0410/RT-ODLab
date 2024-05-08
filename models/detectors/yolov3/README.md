# YOLOv3:

|    Model    |   Backbone   | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|-------------|--------------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| YOLOv3-Tiny | DarkNet-Tiny | 1xb16 |  640  |        25.4            |       43.4        |   7.0             |   2.3              | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolov3_t_coco.pth) |
| YOLOv3      | DarkNet-53   | 1xb16 |  640  |        42.9            |       63.5        |   167.4           |   54.9             | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolov3_coco.pth) |

- For training, we train YOLOv3 and YOLOv3-Tiny with 250 epochs on COCO.
- For data augmentation, we use the large scale jitter (LSJ), Mosaic augmentation and Mixup augmentation, following the setting of [YOLOv5](https://github.com/ultralytics/yolov5).
- For optimizer, we use SGD with momentum 0.937, weight decay 0.0005 and base lr 0.01.
- For learning rate scheduler, we use linear decay scheduler.
- For YOLOv3's structure, we use decoupled head, following the setting of [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX).

## Train YOLOv3
### Single GPU
Taking training YOLOv3 on COCO as the example,
```Shell
python train.py --cuda -d coco --root path/to/coco -m yolov3 -bs 16 -size 640 --wp_epoch 3 --max_epoch 300 --eval_epoch 10 --no_aug_epoch 20 --ema --fp16 --multi_scale 
```

### Multi GPU
Taking training YOLOv3 on COCO as the example,
```Shell
python -m torch.distributed.run --nproc_per_node=8 train.py --cuda -dist -d coco --root /data/datasets/ -m yolov3 -bs 128 -size 640 --wp_epoch 3 --max_epoch 300  --eval_epoch 10 --no_aug_epoch 20 --ema --fp16 --sybn --multi_scale --save_folder weights/ 
```

## Test YOLOv3
Taking testing YOLOv3 on COCO-val as the example,
```Shell
python test.py --cuda -d coco --root path/to/coco -m yolov3 --weight path/to/yolov3_coco.pth -size 640 --show 
```

## Evaluate YOLOv3
Taking evaluating YOLOv3 on COCO-val as the example,
```Shell
python eval.py --cuda -d coco --root path/to/coco -m yolov3 --weight path/to/yolov3_coco.pth
```

## Demo
### Detect with Image
```Shell
python demo.py --mode image --path_to_img path/to/image_dirs/ --cuda -m yolov3 --weight path/to/yolov3_coco.pth -size 640 --show
```

### Detect with Video
```Shell
python demo.py --mode video --path_to_vid path/to/video --cuda -m yolov3 --weight path/to/yolov3_coco.pth -size 640 --show --gif
```

### Detect with Camera
```Shell
python demo.py --mode camera --cuda -m yolov3 --weight path/to/yolov3_coco.pth -size 640 --show --gif
```
