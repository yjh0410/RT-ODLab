# YOLOX:

|   Model | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|---------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| YOLOX-S | 8xb8  |  640  |         40.1           |       60.3        |   26.8            |   8.9              | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolox_s_coco.pth) |
| YOLOX-M | 8xb8  |  640  |         46.2           |       66.0        |   74.3            |   25.4             | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolox_m_coco.pth) |
| YOLOX-L | 8xb8  |  640  |         48.7           |       68.0        |   155.4           |   54.2             | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolox_l_coco.pth) |
| YOLOX-X | 8xb8  |  640  |                        |                   |                   |                    |  |

- For training, we train YOLOX series with 300 epochs on COCO.
- For data augmentation, we use the large scale jitter (LSJ), Mosaic augmentation and Mixup augmentation.
- For optimizer, we use SGD with weight decay 0.0005 and base per image lr 0.01 / 64,.
- For learning rate scheduler, we use Cosine decay scheduler.

## Train YOLOX
### Single GPU
Taking training YOLOX-S on COCO as the example,
```Shell
python train.py --cuda -d coco --root path/to/coco -m yolox_s -bs 16 -size 640 --wp_epoch 3 --max_epoch 300 --eval_epoch 10 --no_aug_epoch 20 --ema --fp16 --multi_scale 
```

### Multi GPU
Taking training YOLOX-S on COCO as the example,
```Shell
python -m torch.distributed.run --nproc_per_node=8 train.py --cuda -dist -d coco --root /data/datasets/ -m yolox_s -bs 128 -size 640 --wp_epoch 3 --max_epoch 300  --eval_epoch 10 --no_aug_epoch 20 --ema --fp16 --sybn --multi_scale --save_folder weights/ 
```

## Test YOLOX
Taking testing YOLOX-S on COCO-val as the example,
```Shell
python test.py --cuda -d coco --root path/to/coco -m yolox_s --weight path/to/yolox_s.pth -size 640 -vt 0.4 --show 
```

## Evaluate YOLOX
Taking evaluating YOLOX-S on COCO-val as the example,
```Shell
python eval.py --cuda -d coco-val --root path/to/coco -m yolox_s --weight path/to/yolox_s.pth 
```

## Demo
### Detect with Image
```Shell
python demo.py --mode image --path_to_img path/to/image_dirs/ --cuda -m yolox_s --weight path/to/weight -size 640 -vt 0.4 --show
```

### Detect with Video
```Shell
python demo.py --mode video --path_to_vid path/to/video --cuda -m yolox_s --weight path/to/weight -size 640 -vt 0.4 --show --gif
```

### Detect with Camera
```Shell
python demo.py --mode camera --cuda -m yolox_s --weight path/to/weight -size 640 -vt 0.4 --show --gif
```