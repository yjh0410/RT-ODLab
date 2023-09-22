# YOLOv7:

|    Model    |   Backbone    | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|-------------|---------------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| YOLOv7-Tiny | ELANNet-Tiny  | 8xb16 |  640  |         39.5           |       58.5        |   22.6            |   7.9              | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolov7_tiny_coco.pth) |
| YOLOv7      | ELANNet-Large | 8xb16 |  640  |         49.5           |       68.8        |   144.6           |   44.0             | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolov7_coco.pth) |
| YOLOv7-X    | ELANNet-Huge  |       |  640  |                        |                   |                   |                    |  |

- For training, we train `YOLOv7` and `YOLOv7-Tiny` with 300 epochs on 8 GPUs.
- For data augmentation, we use the [YOLOX-style](https://github.com/Megvii-BaseDetection/YOLOX) augmentation including the large scale jitter (LSJ), Mosaic augmentation and Mixup augmentation.
- For optimizer, we use `AdamW` with weight decay 0.05 and per image learning rate 0.001 / 64.
- For learning rate scheduler, we use Cosine decay scheduler.
- For YOLOv7's structure, we replace the coupled head with the YOLOX-style decoupled head.
- I think YOLOv7 uses too many training tricks, such as `anchor box`, `AuxiliaryHead`, `RepConv`, `Mosaic9x` and so on, making the picture of YOLO too complicated, which is against the development concept of the YOLO series. Otherwise, why don't we use the DETR series? It's nothing more than doing some acceleration optimization on DETR. Therefore, I was faithful to my own technical aesthetics and realized a cleaner and simpler YOLOv7, but without the blessing of so many tricks, I did not reproduce all the performance, which is a pity.
- I have no more GPUs to train my `YOLOv7-X`.

## Train YOLOv7
### Single GPU
Taking training YOLOv7-Tiny on COCO as the example,
```Shell
python train.py --cuda -d coco --root path/to/coco -m yolov7_tiny -bs 16 -size 640 --wp_epoch 3 --max_epoch 300 --eval_epoch 10 --no_aug_epoch 20 --ema --fp16 --multi_scale 
```

### Multi GPU
Taking training YOLOv7-Tiny on COCO as the example,
```Shell
python -m torch.distributed.run --nproc_per_node=8 train.py --cuda -dist -d coco --root /data/datasets/ -m yolov7_tiny -bs 128 -size 640 --wp_epoch 3 --max_epoch 300  --eval_epoch 10 --no_aug_epoch 20 --ema --fp16 --sybn --multi_scale --save_folder weights/ 
```

## Test YOLOv7
Taking testing YOLOv7-Tiny on COCO-val as the example,
```Shell
python test.py --cuda -d coco --root path/to/coco -m yolov7_tiny --weight path/to/yolov7_tiny.pth -size 640 -vt 0.4 --show 
```

## Evaluate YOLOv7
Taking evaluating YOLOv7-Tiny on COCO-val as the example,
```Shell
python eval.py --cuda -d coco-val --root path/to/coco -m yolov7_tiny --weight path/to/yolov7_tiny.pth 
```

## Demo
### Detect with Image
```Shell
python demo.py --mode image --path_to_img path/to/image_dirs/ --cuda -m yolov7_tiny --weight path/to/weight -size 640 -vt 0.4 --show
```

### Detect with Video
```Shell
python demo.py --mode video --path_to_vid path/to/video --cuda -m yolov7_tiny --weight path/to/weight -size 640 -vt 0.4 --show --gif
```

### Detect with Camera
```Shell
python demo.py --mode camera --cuda -m yolov7_tiny --weight path/to/weight -size 640 -vt 0.4 --show --gif
```
