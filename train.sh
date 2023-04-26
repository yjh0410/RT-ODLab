# Train YOLO on VOC
python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset/ \
        -m yolov7_large \
        -bs 16 \
        -size 640 \
        --wp_epoch 1 \
        --max_epoch 300 \
        --eval_epoch 10 \
        --ema \
        --fp16 \
        --multi_scale \
        --resume weights/coco/yolov7_large/yolov7_large_epoch_121_43.63.pth \
        # --pretrained weights/coco/yolo_free_medium/yolo_free_medium_39.46.pth \
        # --eval_first

