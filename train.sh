# Train YOLO
python train.py \
        --cuda \
        -d voc \
        --root /mnt/share/ssd2/dataset/ \
        -m yolox_n \
        -bs 16 \
        -size 640 \
        --wp_epoch 1 \
        --max_epoch 5 \
        --eval_epoch 1 \
        --no_aug_epoch 2 \
        --ema \
        --fp16 \
        --multi_scale \
        # --resume weights/coco/yolox_m/yolox_m_best.pth \
        # --pretrained weights/coco/yolo_free_medium/yolo_free_medium_39.46.pth \
        # --eval_first


# # Train RT-DETR
# python train.py \
#         --cuda \
#         -d voc \
#         --root /mnt/share/ssd2/dataset/ \
#         -m rtdetr_n \
#         -bs 16 \
#         -size 640 \
#         --wp_epoch 1 \
#         --max_epoch 150 \
#         --eval_epoch 10 \
#         --ema \
#         --fp16 \
#         --multi_scale \
#         --mosaic 0 \
#         --mixup 0
#         # --resume weights/coco/yolox_s/yolox_s_best.pth \
#         # --pretrained weights/coco/yolo_free_medium/yolo_free_medium_39.46.pth \
#         # --eval_first

