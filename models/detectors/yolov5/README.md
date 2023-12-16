# YOLOv5:

|   Model   | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|-----------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| YOLOv5-N  | 1xb16 |  640  |         29.8           |       47.1        |   7.7             |   2.4              | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolov5_n_coco.pth) |
| YOLOv5-S  | 1xb16 |  640  |         37.8           |       56.5        |   27.1            |   9.0              | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolov5_s_coco.pth) |
| YOLOv5-M  | 1xb16 |  640  |         43.5           |       62.5        |   74.3            |   25.4             | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolov5_m_coco.pth) |
| YOLOv5-L  | 1xb16 |  640  |         46.7           |       65.5        |   155.6           |   54.2             | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolov5_l_coco.pth) |

- For training, we train YOLOv5 series with 300 epochs on COCO.
- For data augmentation, we use the large scale jitter (LSJ), Mosaic augmentation and Mixup augmentation, following the setting of [YOLOv5](https://github.com/ultralytics/yolov5).
- For optimizer, we use SGD with weight decay 0.0005 and base per image lr 0.01 / 64, following the setting of the official YOLOv5.
- For learning rate scheduler, we use linear decay scheduler.
- We use decoupled head in our reproduced YOLOv5, which is different from the official YOLOv5'head.


On the other hand, we are trying to use **AdamW** and larger batch size to train our reproduced YOLOv5. We will update the new results as soon as possible.

|   Model   | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|-----------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| YOLOv5-N  | 8xb16 |  640  |                        |                   |                   |                    |  |
| YOLOv5-T  | 8xb16 |  640  |                        |                   |                   |                    |  |
| YOLOv5-S  | 8xb16 |  640  |                        |                   |                   |                    |  |
| YOLOv5-M  | 8xb16 |  640  |                        |                   |                   |                    |  |
| YOLOv5-L  | 8xb16 |  640  |                        |                   |                   |                    |  |
| YOLOv5-X  | 8xb16 |  640  |                        |                   |                   |                    |  |

- For training, we train YOLOv5 series with 300 epochs on COCO.
- For data augmentation, we use the large scale jitter (LSJ), Mosaic augmentation and Mixup augmentation, following the setting of [YOLOv5](https://github.com/ultralytics/yolov5).
- For optimizer, we use AdamW with weight decay 0.05 and base per image lr 0.001 / 64. We are not good at using SGD.
- For learning rate scheduler, we use linear decay scheduler.
- We use decoupled head in our reproduced YOLOv5, which is different from the official YOLOv5'head.


## Train YOLOv5
### Single GPU
Taking training YOLOv5-S on COCO as the example,
```Shell
python train.py --cuda -d coco --root path/to/coco -m yolov5_s -bs 16 -size 640 --wp_epoch 3 --max_epoch 300 --eval_epoch 10 --no_aug_epoch 20 --ema --fp16 --multi_scale 
```

### Multi GPU
Taking training YOLOv5 on COCO as the example,
```Shell
python -m torch.distributed.run --nproc_per_node=8 train.py --cuda -dist -d coco --root /data/datasets/ -m yolov5_s -bs 128 -size 640 --wp_epoch 3 --max_epoch 300  --eval_epoch 10 --no_aug_epoch 20 --ema --fp16 --sybn --multi_scale --save_folder weights/ 
```

## Test YOLOv5
Taking testing YOLOv5 on COCO-val as the example,
```Shell
python test.py --cuda -d coco --root path/to/coco -m yolov5_s --weight path/to/yolov5.pth -size 640 -vt 0.4 --show 
```

## Evaluate YOLOv5
Taking evaluating YOLOv5 on COCO-val as the example,
```Shell
python eval.py --cuda -d coco-val --root path/to/coco -m yolov5_s --weight path/to/yolov5.pth 
```

## Demo
### Detect with Image
```Shell
python demo.py --mode image --path_to_img path/to/image_dirs/ --cuda -m yolov5_s --weight path/to/weight -size 640 -vt 0.4 --show
```

### Detect with Video
```Shell
python demo.py --mode video --path_to_vid path/to/video --cuda -m yolov5_s --weight path/to/weight -size 640 -vt 0.4 --show --gif
```

### Detect with Camera
```Shell
python demo.py --mode camera --cuda -m yolov5_s --weight path/to/weight -size 640 -vt 0.4 --show --gif
```
