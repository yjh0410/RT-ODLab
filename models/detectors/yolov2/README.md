# Redesigned YOLOv2:

| Model  |  Backbone  | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|--------|------------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| YOLOv2 | DarkNet-19 | 1xb16 |  640  |        32.7            |       50.9        |   53.9            |   30.9             | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolov2_coco.pth) |

- For training, we train redesigned YOLOv2 with 150 epochs on COCO.
- For data augmentation, we only use the large scale jitter (LSJ), no Mosaic or Mixup augmentation.
- For optimizer, we use SGD with momentum 0.937, weight decay 0.0005 and base lr 0.01.
- For learning rate scheduler, we use linear decay scheduler.

## Train YOLOv2
### Single GPU
Taking training YOLOv2 on COCO as the example,
```Shell
python train.py --cuda -d coco --root path/to/coco -m yolov2 -bs 16 -size 640 --wp_epoch 3 --max_epoch 200 --eval_epoch 10 --no_aug_epoch 15 --ema --fp16 --multi_scale 
```

### Multi GPU
Taking training YOLOv2 on COCO as the example,
```Shell
python -m torch.distributed.run --nproc_per_node=8 train.py --cuda -dist -d coco --root /data/datasets/ -m yolov2 -bs 128 -size 640 --wp_epoch 3 --max_epoch 200  --eval_epoch 10 --no_aug_epoch 15 --ema --fp16 --sybn --multi_scale --save_folder weights/ 
```

## Test YOLOv2
Taking testing YOLOv2 on COCO-val as the example,
```Shell
python test.py --cuda -d coco --root path/to/coco -m yolov2 --weight path/to/yolov2_coco.pth -size 640 --show 
```

## Evaluate YOLOv2
Taking evaluating YOLOv2 on COCO-val as the example,
```Shell
python eval.py --cuda -d coco --root path/to/coco -m yolov2 --weight path/to/yolov2_coco.pth
```

## Demo
### Detect with Image
```Shell
python demo.py --mode image --path_to_img path/to/image_dirs/ --cuda -m yolov2 --weight path/to/yolov2_coco.pth -size 640 --show
```

### Detect with Video
```Shell
python demo.py --mode video --path_to_vid path/to/video --cuda -m yolov2 --weight path/to/yolov2_coco.pth -size 640 --show --gif
```

### Detect with Camera
```Shell
python demo.py --mode camera --cuda -m yolov2 --weight path/to/yolov2_coco.pth -size 640 --show --gif
```
