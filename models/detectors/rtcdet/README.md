# RTCDet: My Empirical Study of Real-Time Convolutional Object Detectors.

|   Model  | Scale | Batch | AP<sup>test<br>0.5:0.95 | AP<sup>test<br>0.5 | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|----------|-------|-------|-------------------------|--------------------|------------------------|-------------------|-------------------|--------------------|--------|
| RTCDet-N |  640  | 8xb16 |                         |                    |                        |                   |                   |                    |  |
| RTCDet-T |  640  | 8xb16 |                         |                    |                        |                   |                   |                    |  |
| RTCDet-S |  640  | 8xb16 |                         |                    |           44.5         |       63.5        |        30.9       |         8.5        | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/rtcdet_s_coco.pth) |
| RTCDet-M |  640  | 8xb16 |                         |                    |           48.7         |       67.6        |        80.3       |         22.6       | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/rtcdet_m_coco.pth) |
| RTCDet-L |  640  | 8xb16 |                         |                    |                        |                   |                   |                    |  |
| RTCDet-X |  640  | 8xb16 |                         |                    |                        |                   |                   |                    |  |

|   Model  | Scale | Batch | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|----------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| RTCDet-P |  320  | 8xb16 |                        |                   |                   |                    | - |
| RTCDet-P |  416  | 8xb16 |                        |                   |                   |                    | - |
| RTCDet-P |  512  | 8xb16 |                        |                   |                   |                    | - |
| RTCDet-P |  640  | 8xb16 |                        |                   |                   |                    | - |

- For training, we train my RTCDet series series with 300 epochs on COCO.
- For data augmentation, we use the large scale jitter (LSJ), Mosaic augmentation and Mixup augmentation, following the setting of [YOLOX](https://github.com/ultralytics/yolov5), but we remove the rotation transformation which is used in YOLOX's strong augmentation.
- For optimizer, we use AdamW with weight decay 0.05 and base per image lr 0.001 / 64.
- For learning rate scheduler, we use linear decay scheduler.
- Due to my limited computing resources, I can not train `RTCDet-X` with the setting of `batch size=128`.

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
python test.py --cuda -d coco --root path/to/coco -m rtcdet_s --weight path/to/rtcdet_s.pth -size 640 -vt 0.4 --show 
```

## Evaluate RTCDet
Taking evaluating RTCDet-S on COCO-val as the example,
```Shell
python eval.py --cuda -d coco-val --root path/to/coco -m rtcdet_s --weight path/to/rtcdet_s.pth 
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