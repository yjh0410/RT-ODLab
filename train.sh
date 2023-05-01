# Train YOLO
python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset/ \
        -m yolov5_l \
        -bs 16 \
        -size 640 \
        --wp_epoch 1 \
        --max_epoch 300 \
        --eval_epoch 10 \
        --ema \
        --fp16 \
        --multi_scale \
        --resume weights/coco/yolov5_l/yolov5_l_epoch_221_46.56.pth \
        # --pretrained weights/coco/yolo_free_medium/yolo_free_medium_39.46.pth \
        # --eval_first

