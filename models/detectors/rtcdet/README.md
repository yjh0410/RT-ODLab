# RTCDet:

|   Model   |  Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|-----------|--------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| RTCDet-N  | 8xb16  |  640  |                        |                   |                   |                    |  |
| RTCDet-S  | 8xb16  |  640  |                        |                   |                   |                    |  |
| RTCDet-M  | 8xb16  |  640  |                        |                   |                   |                    |  |
| RTCDet-L  | 8xb16  |  640  |                        |                   |                   |                    |  |

- For training, we train RTCDet series with 500 epochs on COCO.
- For data augmentation, we use the large scale jitter (LSJ), Mosaic augmentation and Mixup augmentation, following the setting of [RTCDet](https://github.com/ultralytics/RTCDet).
- For optimizer, we use AdamW with weight decay 0.05 and base per image lr 0.001 / 64, which is different from the official RTCDet. We have tried SGD, but it has weakened performance. For example, when using SGD, RTCDet-N's AP was only 35.8%, lower than the current result (36.8 %), perhaps because some hyperparameters were not set properly.
- For learning rate scheduler, we use linear decay scheduler.


## Train RTCDet
### Single GPU
Taking training RTCDet-S on COCO as the example,
```Shell
python train.py --cuda -d coco --root path/to/coco -m rtcdet_s -bs 16 -size 640 --wp_epoch 3 --max_epoch 500 --eval_epoch 10 --no_aug_epoch 20 --ema --fp16 --multi_scale 
```

### Multi GPU
Taking training RTCDet on COCO as the example,
```Shell
python -m torch.distributed.run --nproc_per_node=8 train.py --cuda -dist -d coco --root /data/datasets/ -m rtcdet_s -bs 128 -size 640 --wp_epoch 3 --max_epoch 500  --eval_epoch 10 --no_aug_epoch 20 --ema --fp16 --sybn --multi_scale --save_folder weights/ 
```

## Test RTCDet
Taking testing RTCDet on COCO-val as the example,
```Shell
python test.py --cuda -d coco --root path/to/coco -m rtcdet_s --weight path/to/RTCDet.pth -size 640 -vt 0.4 --show 
```

## Evaluate RTCDet
Taking evaluating RTCDet on COCO-val as the example,
```Shell
python eval.py --cuda -d coco-val --root path/to/coco -m rtcdet_s --weight path/to/RTCDet.pth 
```

## Demo
### Detect with Image
```Shell
python demo.py --mode image --path_to_img path/to/image_dirs/ --cuda -m rtcdet_s --weight path/to/weight -size 640 -vt 0.4 --show
```

### Detect with Video
```Shell
python demo.py --mode video --path_to_vid path/to/video --cuda -m rtcdet_s --weight path/to/weight -size 640 -vt 0.4 --show --gif
```

### Detect with Camera
```Shell
python demo.py --mode camera --cuda -m rtcdet_s --weight path/to/weight -size 640 -vt 0.4 --show --gif
```
