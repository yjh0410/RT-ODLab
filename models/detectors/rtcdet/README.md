# RTCDet: My Empirical Study of Real-Time Convolutional Object Detectors.

|   Model  | Scale | Batch | AP<sup>test<br>0.5:0.95 | AP<sup>test<br>0.5 | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|----------|-------|-------|-------------------------|--------------------|------------------------|-------------------|-------------------|--------------------|--------|
| RTCDet-N |  640  | 8xb16 |                         |                    |                        |                   |                   |                    |  |
| RTCDet-T |  640  | 8xb16 |                         |                    |                        |                   |                   |                    |  |
| RTCDet-S |  640  | 8xb16 |                         |                    |                        |                   |                   |                    |  |
| RTCDet-M |  640  | 8xb16 |                         |                    |           48.7         |       67.6        |        80.3       |          22.6      |  |
| RTCDet-L |  640  | 8xb16 |                         |                    |                        |                   |                   |                    |  |
| RTCDet-X |  640  | 8xb16 |                         |                    |                        |                   |                   |                    |  |

|   Model  | Scale | Batch | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|----------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| RTCDet-P |  320  | 8xb16 |                        |                   |                   |                    | - |
| RTCDet-P |  416  | 8xb16 |                        |                   |                   |                    | - |
| RTCDet-P |  512  | 8xb16 |                        |                   |                   |                    | - |
| RTCDet-P |  640  | 8xb16 |                        |                   |                   |                    | - |

- For training, we train my RTCDet series series with 300 epochs on COCO.
- For data augmentation, we use the large scale jitter (LSJ), Mosaic augmentation and Mixup augmentation, following the setting of [YOLOX](https://github.com/ultralytics/yolov5), but we remove the rotation transformation which is used in YOLOX's strong augmentation.
- For optimizer, we use AdamW with weight decay 0.05 and base per image lr 0.001 / 64.
- For learning rate scheduler, we use linear decay scheduler.
- Due to my limited computing resources, I can not train `RTCDet-X` with the setting of `batch size=128`.
