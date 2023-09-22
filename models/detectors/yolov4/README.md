# YOLOv4:

|    Model    |     Backbone    | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|-------------|-----------------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| YOLOv4-Tiny | CSPDarkNet-Tiny | 1xb16 |  640  |        31.0            |       49.1        |   8.1             |   2.9              | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolov4_t_coco.pth) |
| YOLOv4      | CSPDarkNet-53   | 1xb16 |  640  |        46.6            |       65.8        |   162.7           |   61.5             | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolov4_coco.pth) |

- For training, we train YOLOv4 and YOLOv4-Tiny with 250 epochs on COCO.
- For data augmentation, we use the large scale jitter (LSJ), Mosaic augmentation and Mixup augmentation, following the setting of [YOLOv5](https://github.com/ultralytics/yolov5).
- For optimizer, we use SGD with momentum 0.937, weight decay 0.0005 and base lr 0.01.
- For learning rate scheduler, we use linear decay scheduler.
- For YOLOv4's structure, we use decoupled head, following the setting of [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX).

## Train YOLOv4
### Single GPU
Taking training YOLOv4 on COCO as the example,
```Shell
python train.py --cuda -d coco --root path/to/coco -m yolov4 -bs 16 -size 640 --wp_epoch 3 --max_epoch 300 --eval_epoch 10 --no_aug_epoch 20 --ema --fp16 --multi_scale 
```

### Multi GPU
Taking training YOLOv4 on COCO as the example,
```Shell
python -m torch.distributed.run --nproc_per_node=8 train.py --cuda -dist -d coco --root /data/datasets/ -m yolov4 -bs 128 -size 640 --wp_epoch 3 --max_epoch 300  --eval_epoch 10 --no_aug_epoch 20 --ema --fp16 --sybn --multi_scale --save_folder weights/ 
```

## Test YOLOv4
Taking testing YOLOv4 on COCO-val as the example,
```Shell
python test.py --cuda -d coco --root path/to/coco -m yolov4 --weight path/to/yolov4.pth -size 640 -vt 0.4 --show 
```

## Evaluate YOLOv4
Taking evaluating YOLOv4 on COCO-val as the example,
```Shell
python eval.py --cuda -d coco-val --root path/to/coco -m yolov4 --weight path/to/yolov4.pth 
```

## Demo
### Detect with Image
```Shell
python demo.py --mode image --path_to_img path/to/image_dirs/ --cuda -m yolov4 --weight path/to/weight -size 640 -vt 0.4 --show
```

### Detect with Video
```Shell
python demo.py --mode video --path_to_vid path/to/video --cuda -m yolov4 --weight path/to/weight -size 640 -vt 0.4 --show --gif
```

### Detect with Camera
```Shell
python demo.py --mode camera --cuda -m yolov4 --weight path/to/weight -size 640 -vt 0.4 --show --gif
```
