# Real-time Transformer-based Object Detector:
This model is not yet complete.

## Results on the COCO-val
|     Model    | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|--------------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| RT-DETR-R18  | 4xb4  |  640  |                        |                   |                   |                    |  |
| RT-DETR-R50  | 4xb4  |  640  |                        |                   |                   |                    |  |
| RT-DETR-R101 | 4xb4  |  640  |                        |                   |                   |                    |  |

- For the backbone of the image encoder, we use the IN-1K classification pretrained weight. It might be hard to train RT-DETR from scratch without IN-1K pretrained weight.
- For training, we train RT-DETR series with 6x (~72 epochs) schedule on COCO.
- For data augmentation, we use the `color jitter`, `random hflip`, `random crop`, and multi-scale training trick.
- For optimizer, we use AdamW with weight decay 0.0001 and base per image lr 0.001 / 16.
- For learning rate scheduler, we use `cosine` decay scheduler.

## Train RT-DETR
### Single GPU
Taking training RT-DETR-R18 on COCO as the example,
```Shell
python train.py --cuda -d coco --root path/to/coco -m rtdetr_r18 -bs 16 -size 640 --max_epoch 72 --eval_epoch 5 --no_aug_epoch -1 --ema --fp16 --multi_scale 
```

### Multi GPU
Taking training RT-DETR-R18 on COCO as the example,
```Shell
python -m torch.distributed.run --nproc_per_node=8 train.py --cuda -dist -d coco --root /data/datasets/ -m rtdetr_r18 -bs 16 -size 640 --max_epoch 72 --eval_epoch 5 --no_aug_epoch -1 --ema --fp16 --sybn --multi_scale --save_folder weights/ 
```

## Test RT-DETR
Taking testing RT-DETR-R18 on COCO-val as the example,
```Shell
python test.py --cuda -d coco --root path/to/coco -m rtdetr_r18 --weight path/to/rtdetr_r18.pth -size 640 -vt 0.4 --show 
```

## Evaluate RT-DETR
Taking evaluating RT-DETR-R18 on COCO-val as the example,
```Shell
python eval.py --cuda -d coco-val --root path/to/coco -m rtdetr_r18 --weight path/to/rtdetr_r18.pth 
```

## Demo
### Detect with Image
```Shell
python demo.py --mode image --path_to_img path/to/image_dirs/ --cuda -m rtdetr_r18 --weight path/to/weight -size 640 -vt 0.4 --show
```

### Detect with Video
```Shell
python demo.py --mode video --path_to_vid path/to/video --cuda -m rtdetr_r18 --weight path/to/weight -size 640 -vt 0.4 --show --gif
```

### Detect with Camera
```Shell
python demo.py --mode camera --cuda -m rtdetr_r18 --weight path/to/weight -size 640 -vt 0.4 --show --gif
```