# RTCDet-v1: My First Empirical Study of Real-Time Convolutional Object Detectors.

|   Model    | Scale | Batch | AP<sup>test<br>0.5:0.95 | AP<sup>test<br>0.5 | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|------------|-------|-------|-------------------------|--------------------|------------------------|-------------------|-------------------|--------------------|--------|
| RTCDetv1-N |  640  | 8xb16 |         35.7            |        53.8        |          35.6          |        53.8       |      9.1          |        2.4         | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/rtcdet_v1_n_coco.pth) |
| RTCDetv1-T |  640  | 8xb16 |         40.5            |        59.1        |          40.3          |        59.1       |      19.0         |        5.1         | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/rtcdet_v1_t_coco.pth) |
| RTCDetv1-S |  640  | 8xb16 |         43.6            |        62.6        |          43.3          |        62.6       |      33.6         |        9.0         | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/rtcdet_v1_s_coco.pth) |
| RTCDetv1-M |  640  | 8xb16 |         48.3            |        67.0        |          48.1          |        66.9       |      87.4         |        23.6        | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/rtcdet_v1_m_coco.pth) |
| RTCDetv1-L |  640  | 8xb16 |         50.2            |        68.6        |          50.0          |        68.4       |      176.6        |        47.6        | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/rtcdet_v1_l_coco.pth) |
| RTCDetv1-X |  640  | 8xb12 |                         |                    |                        |                   |                   |                    |  |

- For training, we train my RTCDetv1 series series with 300 epochs on COCO.
- For data augmentation, we use the large scale jitter (LSJ), Mosaic augmentation and Mixup augmentation, following the setting of [YOLOX](https://github.com/ultralytics/yolov5), but we remove the rotation transformation which is used in YOLOX's strong augmentation.
- For optimizer, we use AdamW with weight decay 0.05 and base per image lr 0.001 / 64.
- For learning rate scheduler, we use linear decay scheduler.
- Due to my limited computing resources, I can not train `RTCDetv1-X` with the setting of `batch size=128`.
