# YOLOv3:

|    Model    |   Backbone   | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|-------------|--------------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| YOLOv3-Tiny | DarkNet-Tiny | 1xb16 |  640  |        25.4            |       43.4        |   7.0             |   2.3              | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov3_t_coco.pth) |
| YOLOv3      | DarkNet-53   | 1xb16 |  640  |        42.9            |       63.5        |   167.4           |   54.9             | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov3_coco.pth) |

- For training, we train YOLOv3 and YOLOv3-Tiny with 250 epochs on COCO.
- For data augmentation, we use the large scale jitter (LSJ), Mosaic augmentation and Mixup augmentation, following the setting of [YOLOv5](https://github.com/ultralytics/yolov5).
- For optimizer, we use SGD with momentum 0.937, weight decay 0.0005 and base lr 0.01.
- For learning rate scheduler, we use linear decay scheduler.
- For YOLOv3's structure, we use decoupled head, following the setting of [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX).