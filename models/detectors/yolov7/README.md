# YOLOv7:

|    Model    |   Backbone    | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|-------------|---------------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| YOLOv7-Tiny | ELANNet-Tiny  | 8xb16 |  640  |         39.5           |       58.5        |   22.6            |   7.9              | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov7_tiny_coco.pth) |
| YOLOv7      | ELANNet-Large | 8xb16 |  640  |         49.5           |       68.8        |   144.6           |   44.0             | [ckpt](https://github.com/yjh0410/PyTorch_YOLO_Tutorial/releases/download/yolo_tutorial_ckpt/yolov7_coco.pth) |
| YOLOv7-X    | ELANNet-Huge  |       |  640  |                        |                   |                   |                    |  |

- For training, we train `YOLOv7` and `YOLOv7-Tiny` with 300 epochs on 8 GPUs.
- For data augmentation, we use the [YOLOX-style](https://github.com/Megvii-BaseDetection/YOLOX) augmentation including the large scale jitter (LSJ), Mosaic augmentation and Mixup augmentation.
- For optimizer, we use `AdamW` with weight decay 0.05 and per image learning rate 0.001 / 64.
- For learning rate scheduler, we use Cosine decay scheduler.
- For YOLOv7's structure, we replace the coupled head with the YOLOX-style decoupled head.
- I think YOLOv7 uses too many training tricks, such as `anchor box`, `AuxiliaryHead`, `RepConv`, `Mosaic9x` and so on, making the picture of YOLO too complicated, which is against the development concept of the YOLO series. Otherwise, why don't we use the DETR series? It's nothing more than doing some acceleration optimization on DETR. Therefore, I was faithful to my own technical aesthetics and realized a cleaner and simpler YOLOv7, but without the blessing of so many tricks, I did not reproduce all the performance, which is a pity.
- I have no more GPUs to train my `YOLOv7-X`.