# YOLOX:

|   Model |   Backbone   | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|---------|--------------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| YOLOX-S | CSPDarkNet-S | 8xb8  |  640  |         40.1           |       60.3        |   26.8            |   8.9              | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolox_s_coco.pth) |
| YOLOX-M | CSPDarkNet-M | 8xb8  |  640  |         46.2           |       66.0        |   74.3            |   25.4             | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolox_m_coco.pth) |
| YOLOX-L | CSPDarkNet-L | 8xb8  |  640  |         48.7           |       68.0        |   155.4           |   54.2             | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolox_l_coco.pth) |
| YOLOX-X | CSPDarkNet-X | 8xb8  |  640  |                        |                   |                   |                    |  |

- For training, we train YOLOX series with 300 epochs on COCO.
- For data augmentation, we use the large scale jitter (LSJ), Mosaic augmentation and Mixup augmentation.
- For optimizer, we use SGD with weight decay 0.0005 and base per image lr 0.01 / 64,.
- For learning rate scheduler, we use Cosine decay scheduler.
- The reason for the low performance of my reproduced **YOLOX-L** has not been found out yet.