# Train FreeYOLO
python train.py \
        --cuda \
        -d voc \
        --root /mnt/share/ssd2/dataset/ \
        -m yolov2 \
        -bs 16 \
        -size 640 \
        --wp_epoch 1 \
        --max_epoch 150 \
        --eval_epoch 10 \
        --ema \
        --fp16 \
        --multi_scale \
        # --resume weights/coco/yolo_free_vx_pico/yolo_free_vx_pico_epoch_41_20.46.pth \
        # --pretrained weights/coco/yolo_free_medium/yolo_free_medium_39.46.pth \
        # --eval_first
