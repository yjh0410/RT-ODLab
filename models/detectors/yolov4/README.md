# YOLOv4:

|    Model    |     Backbone    | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|-------------|-----------------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| YOLOv4-Tiny | CSPDarkNet-Tiny | 1xb16 |  640  |        31.0            |       49.1        |   8.1             |   2.9              | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov4_t_coco.pth) |
| YOLOv4      | CSPDarkNet-53   | 1xb16 |  640  |        46.6            |       65.8        |   162.7           |   61.5             | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov4_coco.pth) |

- For training, we train YOLOv4 and YOLOv4-Tiny with 250 epochs on COCO.
- For data augmentation, we use the large scale jitter (LSJ), Mosaic augmentation and Mixup augmentation, following the setting of [YOLOv5](https://github.com/ultralytics/yolov5).
- For optimizer, we use SGD with momentum 0.937, weight decay 0.0005 and base lr 0.01.
- For learning rate scheduler, we use linear decay scheduler.
- For YOLOv4's structure, we use decoupled head, following the setting of [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX).