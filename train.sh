# Train YOLO
python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset/ \
        -m yolox_s \
        -bs 16 \
        -size 640 \
        --wp_epoch 3 \
        --max_epoch 300 \
        --eval_epoch 10 \
        --ema \
        --fp16 \
        --multi_scale \
        # --resume weights/coco/yolox_n/yolox_n_best.pth \
        # --pretrained weights/coco/yolo_free_medium/yolo_free_medium_39.46.pth \
        # --eval_first

