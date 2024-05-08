# Redesigned YOLOv1:

| Model  |  Backbone  | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|--------|------------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| YOLOv1 | ResNet-18  | 1xb16 |  640  |        27.9            |       47.5        |   37.8            |   21.3             | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolov1_coco.pth) |

- For training, we train redesigned YOLOv1 with 150 epochs on COCO.
- For data augmentation, we only use the large scale jitter (LSJ), no Mosaic or Mixup augmentation.
- For optimizer, we use SGD with momentum 0.937, weight decay 0.0005 and base lr 0.01.
- For learning rate scheduler, we use linear decay scheduler.


## Train YOLOv1
### Single GPU
Taking training YOLOv1 on COCO as the example,
```Shell
python train.py --cuda -d coco --root path/to/coco -m yolov1 -bs 16 -size 640 --wp_epoch 3 --max_epoch 150 --eval_epoch 10 --no_aug_epoch 10 --ema --fp16 --multi_scale 
```

### Multi GPU
Taking training YOLOv1 on COCO as the example,
```Shell
python -m torch.distributed.run --nproc_per_node=8 train.py --cuda -dist -d coco --root /data/datasets/ -m yolov1 -bs 128 -size 640 --wp_epoch 3 --max_epoch 150  --eval_epoch 10 --no_aug_epoch 20 --ema --fp16 --sybn --multi_scale --save_folder weights/ 
```

## Test YOLOv1
Taking testing YOLOv1 on COCO-val as the example,
```Shell
python test.py --cuda -d coco --root path/to/coco -m yolov1 --weight path/to/yolov1_coco.pth -size 640 --show 
```

## Evaluate YOLOv1
Taking evaluating YOLOv1 on COCO-val as the example,
```Shell
python eval.py --cuda -d coco --root path/to/coco -m yolov1 --weight path/to/yolov1_coco.pth 
```

## Demo
### Detect with Image
```Shell
python demo.py --mode image --path_to_img path/to/image_dirs/ --cuda -m yolov1 --weight path/to/yolov1_coco.pth -size 640 --show
```

### Detect with Video
```Shell
python demo.py --mode video --path_to_vid path/to/video --cuda -m yolov1 --weight path/to/yolov1_coco.pth -size 640 --show --gif
```

### Detect with Camera
```Shell
python demo.py --mode camera --cuda -m yolov1 --weight path/to/yolov1_coco.pth -size 640 --show --gif
```
