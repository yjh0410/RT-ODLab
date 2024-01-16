# RTCDet:

## Effectiveness of the pretrained weight
- **IN1K Cls**: We pretrained the backbone (RTCNet) on the ImageNet-1K dataset with the classification task setting.
- **IN1K MIM**: We pretrained the backbone (RTCNet) on the ImageNet-1K dataset with the masked image modeling task setting.
- **Scratch**:  We just train the detector on the COCO without any pretrained weights for the backbone.

For the small model:
|   Model  | Pretrained | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|----------|------------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| RTCDet-S | Scratch    |  640  |                        |                   |                   |                    |  |
| RTCDet-S | IN1K Cls   |  640  |                        |                   |                   |                    |  |
| RTCDet-S | IN1K MIM   |  640  |                        |                   |                   |                    |  |

For the large model:
|   Model  | Pretrained | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|----------|------------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| RTCDet-L | Scratch    |  640  |                        |                   |                   |                    |  |
| RTCDet-L | IN1K Cls   |  640  |                        |                   |                   |                    |  |
| RTCDet-L | IN1K MIM   |  640  |                        |                   |                   |                    |  |


## Results on the COCO-val
|   Model  | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|----------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| RTCDet-N | 8xb16 |  640  |                        |                   |                   |                    |  |
| RTCDet-S | 8xb16 |  640  |                        |                   |                   |                    |  |
| RTCDet-M | 8xb16 |  640  |                        |                   |                   |                    |  |
| RTCDet-L | 8xb16 |  640  |                        |                   |                   |                    |  |
| RTCDet-X | 8xb16 |  640  |                        |                   |                   |                    |  |

- For the backbone, we ... (not sure)
- For training, we train RTCDet series with 300 epochs on COCO.
- For data augmentation, we use the large scale jitter (LSJ), Mosaic augmentation and Mixup augmentation, following the YOLOX.
- For optimizer, we use AdamW with weight decay 0.05 and base per image lr 0.001 / 64,.
- For learning rate scheduler, we use Linear decay scheduler.

## Train RTCDet
### Single GPU
Taking training RTCDet-S on COCO as the example,
```Shell
python train.py --cuda -d coco --root path/to/coco -m rtcdet_s -bs 16 -size 640 --wp_epoch 3 --max_epoch 300 --eval_epoch 10 --no_aug_epoch 20 --ema --fp16 --multi_scale 
```

### Multi GPU
Taking training RTCDet-S on COCO as the example,
```Shell
python -m torch.distributed.run --nproc_per_node=8 train.py --cuda -dist -d coco --root /data/datasets/ -m rtcdet_s -bs 128 -size 640 --wp_epoch 3 --max_epoch 300  --eval_epoch 10 --no_aug_epoch 20 --ema --fp16 --sybn --multi_scale --save_folder weights/ 
```

## Test RTCDet
Taking testing RTCDet-S on COCO-val as the example,
```Shell
python test.py --cuda -d coco --root path/to/coco -m rtcdet_s --weight path/to/RTCDet_s.pth -size 640 -vt 0.4 --show 
```

## Evaluate RTCDet
Taking evaluating RTCDet-S on COCO-val as the example,
```Shell
python eval.py --cuda -d coco-val --root path/to/coco -m rtcdet_s --weight path/to/RTCDet_s.pth 
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