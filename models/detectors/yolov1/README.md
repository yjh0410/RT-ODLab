# Redesigned YOLOv1:

| Model  |  Backbone  | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|--------|------------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| YOLOv1 | ResNet-18  | 1xb16 |  640  |        27.9            |       47.5        |   37.8            |   21.3             | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov1_coco.pth) |

- For training, we train redesigned YOLOv1 with 150 epochs on COCO.
- For data augmentation, we only use the large scale jitter (LSJ), no Mosaic or Mixup augmentation.
- For optimizer, we use SGD with momentum 0.937, weight decay 0.0005 and base lr 0.01.
- For learning rate scheduler, we use linear decay scheduler.
