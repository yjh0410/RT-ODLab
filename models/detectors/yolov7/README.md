# YOLOv4:

|    Model    |   Backbone    | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|-------------|---------------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| YOLOv7-Tiny | ELANNet-Tiny  | 8xb16 |  640  |         39.5           |       58.5        |   22.6            |   7.9              | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov7_tiny_coco.pth) |
| YOLOv7      | ELANNet-Large | 1xb16 |  640  |         48.0           |       67.5        |   144.6           |   44.0             | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov7_large_coco.pth) |
| YOLOv7-X    | ELANNet-Huge  | 8xb8  |  640  |                        |                   |                   |                    |  |

- For training, we train YOLOv7 and YOLOv7-Tiny with 300 epochs on COCO.
- For data augmentation, we use the large scale jitter (LSJ), Mosaic augmentation and Mixup augmentation, following the setting of [YOLOv5](https://github.com/ultralytics/yolov5).
- For optimizer, we use SGD with momentum 0.937, weight decay 0.0005 and base lr 0.01.
- For learning rate scheduler, we use linear decay scheduler.
- For YOLOv7's structure, we use decoupled head, following the setting of [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX).
- While YOLOv7 incorporates several technical details, such as anchor box, SimOTA, AuxiliaryHead, and RepConv, I found it too challenging to fully reproduce. Instead, I created a simpler version of YOLOv7 using an anchor-free structure and SimOTA. As a result, my reproduction had poor performance due to the absence of the other technical details. However, since it was only intended as a tutorial, I am not too concerned about this gap.
