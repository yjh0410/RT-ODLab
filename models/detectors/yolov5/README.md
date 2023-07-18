# YOLOv5:

|   Model   |   Backbone   | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|-----------|--------------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| YOLOv5-N  | CSPDarkNet-N | 1xb16 |  640  |         29.8           |       47.1        |   7.7             |   2.4              | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov5_n_coco.pth) |
| YOLOv5-S  | CSPDarkNet-S | 1xb16 |  640  |         37.8           |       56.5        |   27.1            |   9.0              | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov5_s_coco.pth) |
| YOLOv5-M  | CSPDarkNet-M | 1xb16 |  640  |         43.5           |       62.5        |   74.3            |   25.4             | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov5_m_coco.pth) |
| YOLOv5-L  | CSPDarkNet-L | 1xb16 |  640  |         46.7           |       65.5        |   155.6           |   54.2             | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov5_l_coco.pth) |

- For training, we train YOLOv5 series with 300 epochs on COCO.
- For data augmentation, we use the large scale jitter (LSJ), Mosaic augmentation and Mixup augmentation, following the setting of [YOLOv5](https://github.com/ultralytics/yolov5).
- For optimizer, we use SGD with momentum 0.937, weight decay 0.0005 and base lr 0.01.
- For learning rate scheduler, we use linear decay scheduler.
- For YOLOv5's structure, we use decoupled head, following the setting of [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX).
- For **YOLOv5-M** and **YOLOv5-L**, increasing the batch size may improve performance. Due to my computing resources, I can only set the batch size to 16.
