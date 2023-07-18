# YOLOv5:

|   Model |   Backbone   | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|---------|--------------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| YOLOX-N | CSPDarkNet-N | 8xb8  |  640  |         30.4           |       48.9        |   7.5             |   2.3              | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolox_n_coco.pth) |
| YOLOX-S | CSPDarkNet-S | 8xb8  |  640  |         39.0           |       58.8        |   26.8            |   8.9              | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolox_s_coco.pth) |
| YOLOX-M | CSPDarkNet-M | 1xb16 |  640  |         44.6           |       63.8        |   74.3            |   25.4             | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolox_m_coco.pth) |
| YOLOX-L | CSPDarkNet-L | 1xb16 |  640  |         46.9           |       65.9        |   155.4           |   54.2             | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolox_l_coco.pth) |

- For training, we train YOLOX series with 300 epochs on COCO.
- For data augmentation, we use the large scale jitter (LSJ), Mosaic augmentation and Mixup augmentation, following the setting of [YOLOX](https://github.com/ultralytics/yolov5).
- For optimizer, we use AdamW with weight decay 0.05 and base per image lr 0.001 / 64.
- For learning rate scheduler, we use linear decay scheduler.
- I am trying to retrain **YOLOX-M** and **YOLOX-L** with more GPUs, and I will update the AP of YOLOX-M and YOLOX-L in the table in the future.
