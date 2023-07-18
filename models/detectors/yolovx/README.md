# YOLOv5:

| Model    | Scale | Batch | AP<sup>test<br>0.5:0.95 | AP<sup>test<br>0.5 | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|----------|-------|-------|-------------------------|--------------------|------------------------|-------------------|-------------------|--------------------|--------|
| YOLOvx-N |  640  | 8xb16 |                         |                    |                        |                   |      9.1          |        2.4         |  |
| YOLOvx-T |  640  | 8xb16 |                         |                    |                        |                   |      18.9         |        5.1         |  |
| YOLOvx-S |  640  | 8xb16 |                         |                    |                        |                   |      33.6         |        9.0         |  |
| YOLOvx-M |  640  | 8xb16 |         48.3            |        67.0        |          48.1          |        66.9       |      87.4         |        23.6        | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolovx_m_coco.pth) |
| YOLOvx-L |  640  | 8xb16 |         50.2            |        68.6        |          50.0          |        68.4       |      176.6        |        47.6        | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolovx_l_coco.pth) |
| YOLOvx-X |  640  |       |                         |                    |                        |                   |                   |                    |  |

- For training, we train my YOLOvx series series with 300 epochs on COCO.
- For data augmentation, we use the large scale jitter (LSJ), Mosaic augmentation and Mixup augmentation, following the setting of [YOLOX](https://github.com/ultralytics/yolov5), but we remove the rotation transformation which is used in YOLOX's strong augmentation.
- For optimizer, we use AdamW with weight decay 0.05 and base per image lr 0.001 / 64.
- For learning rate scheduler, we use linear decay scheduler.
- Due to my limited computing resources, I can not train `YOLOvx-X` with the setting of `batch size=128`.
