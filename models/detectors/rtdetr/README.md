# Real-time Transformer-based Object Detector:
This model is not yet complete.

## Results on the COCO-val
|     Model    | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight | Los |
|--------------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|-----|
| RT-DETR-R18  | 4xb4  |  640  |           45.5         |        63.0       |        66.8       |        21.0        | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/detr_series_ckpt/rtdetr_r18_coco.pth) | [log](https://github.com/yjh0410/RT-ODLab/releases/download/detr_series_ckpt/RT-DETR-R18-COCO.txt)|
| RT-DETR-R50  | 4xb4  |  640  |           50.2         |        68.5       |       113.7       |        40.4        | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/detr_series_ckpt/rtdetr_r50_coco.pth) | [log](https://github.com/yjh0410/RT-ODLab/releases/download/detr_series_ckpt/RT-DETR-R50-COCO.txt)|
| RT-DETR-R101 | 4xb4  |  640  |                        |                   |                   |                    |  | |

- For the backbone of the image encoder, we use the IN-1K classification pretrained weight from torchvision, which is different from the official
RT-DETR. It might be hard to train RT-DETR from scratch without IN-1K pretrained weight.
- For the HybridEncoder, we use the C2f of YOLOv8 rather than the CSPRepLayer.
- For training, we train RT-DETR series with 6x (~72 epochs) schedule on COCO and use ModelEMA trick. We close the fp16 training trick.
- For data augmentation, we use the `color jitter`, `random hflip`, `random crop`, and multi-scale training trick.
- For optimizer, we use AdamW with weight decay 0.0001 and base per image lr 0.0001 / 16.
- For learning rate scheduler, we use constant learning rate (=0.0001), following the official setting.
- For post-processing, we think it is still a little helpful to deploy NMS even if it is not essential.

## Train RT-DETR
### Single GPU
Taking training RT-DETR-R18 on COCO as the example,
```Shell
python train.py --cuda -d coco --root path/to/coco -m rtdetr_r18 -bs 16 -size 640 --max_epoch 72 --eval_epoch 1 --ema --multi_scale 
```

### Multi GPU
Taking training RT-DETR-R18 on COCO with 4 GPUs as the example,
```Shell
python -m torch.distributed.run --nproc_per_node=4 train.py --cuda -dist -d coco --root /data/datasets/ -m rtdetr_r18 -bs 16 -size 640 --max_epoch 72 --eval_epoch 1 --ema --sybn --multi_scale 
```

## Test RT-DETR
Taking testing RT-DETR-R18 on COCO-val as the example,
```Shell
python test.py --cuda -d coco --root path/to/coco -m rtdetr_r18 --weight path/to/rtdetr_r18.pth -size 640 -ct 0.4 --show 
```

## Evaluate RT-DETR
Taking evaluating RT-DETR-R18 on COCO-val as the example,
```Shell
python eval.py --cuda -d coco --root path/to/coco -m rtdetr_r18 --weight path/to/rtdetr_r18.pth -size 640
```

## Demo
### Detect with Image
```Shell
python demo.py --mode image --path_to_img path/to/image_dirs/ --cuda -m rtdetr_r18 --weight path/to/weight -size 640 -ct 0.4 --show
```

### Detect with Video
```Shell
python demo.py --mode video --path_to_vid path/to/video --cuda -m rtdetr_r18 --weight path/to/weight -size 640 -ct 0.4 --show --gif
```

### Detect with Camera
```Shell
python demo.py --mode camera --cuda -m rtdetr_r18 --weight path/to/weight -size 640 -ct 0.4 --show --gif
```