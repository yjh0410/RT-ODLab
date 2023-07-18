# Redesigned YOLOv2:

| Model  |  Backbone  | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|--------|------------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| YOLOv2 | DarkNet-19 | 1xb16 |  640  |        32.7            |       50.9        |   53.9            |   30.9             | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov2_coco.pth) |

- For training, we train redesigned YOLOv2 with 150 epochs on COCO.
- For data augmentation, we only use the large scale jitter (LSJ), no Mosaic or Mixup augmentation.
- For optimizer, we use SGD with momentum 0.937, weight decay 0.0005 and base lr 0.01.
- For learning rate scheduler, we use linear decay scheduler.
