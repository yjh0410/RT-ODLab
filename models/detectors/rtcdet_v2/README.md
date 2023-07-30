# RTCDet-v2: My Second Empirical Study of Real-Time Convolutional Object Detectors.

|   Model    | Scale | Batch | AP<sup>test<br>0.5:0.95 | AP<sup>test<br>0.5 | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|------------|-------|-------|-------------------------|--------------------|------------------------|-------------------|-------------------|--------------------|--------|
| RTCDetv2-N |  640  | 4xb16 |                         |                    |                        |                   |                   |                    |  |
| RTCDetv2-T |  640  | 4xb16 |                         |                    |                        |                   |                   |                    |  |
| RTCDetv2-S |  640  | 4xb16 |                         |                    |                        |                   |                   |                    |  |
| RTCDetv2-M |  640  | 4xb16 |                         |                    |                        |                   |                   |                    |  |
| RTCDetv2-L |  640  | 4xb16 |                         |                    |                        |                   |                   |                    |  |
| RTCDetv2-X |  640  |       |                         |                    |                        |                   |                   |                    |  |

|   Model    | Scale | Batch | AP<sup>test<br>0.5:0.95 | AP<sup>test<br>0.5 | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|------------|-------|-------|-------------------------|--------------------|------------------------|-------------------|-------------------|--------------------|--------|
| RTCDetv2-P |  320  | 4xb16 |                         |                    |                        |                   |                   |                    |  |
| RTCDetv2-P |  416  | 4xb16 |                         |                    |                        |                   |                   |                    |  |
| RTCDetv2-P |  512  | 4xb16 |                         |                    |                        |                   |                   |                    |  |
| RTCDetv2-P |  640  | 4xb16 |                         |                    |                        |                   |                   |                    |  |

- For training, we train my RTCDetv2 series series with 300 epochs on COCO.
- For data augmentation, we use the large scale jitter (LSJ), Mosaic augmentation and Mixup augmentation, following the setting of [YOLOX](https://github.com/ultralytics/yolov5), but we remove the rotation transformation which is used in YOLOX's strong augmentation.
- For optimizer, we use AdamW with weight decay 0.05 and base per image lr 0.001 / 64.
- For learning rate scheduler, we use linear decay scheduler.
- Due to my limited computing resources, I can not train `RTCDetv2-X` with the setting of `batch size=128`.
