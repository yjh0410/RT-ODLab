# Train FreeYOLO
python train.py \
        --cuda \
        -d voc \
        --root /mnt/share/ssd2/dataset/ \
        -m yolov1 \
        -bs 16 \
        -size 640 \
        --wp_epoch 1 \
        --max_epoch 4 \
        --step_epoch 2 3 \
        --eval_epoch 1 \
        --ema \
        --fp16 \
        --multi_scale \
        # --resume weights/coco/yolo_free_vx_pico/yolo_free_vx_pico_epoch_41_20.46.pth \
        # --pretrained weights/coco/yolo_free_medium/yolo_free_medium_39.46.pth \
        # --eval_first

# # Debug FreeYOLO on VOC
# python train.py \
#         --cuda \
#         -d voc \
#         --root /mnt/share/ssd2/dataset/ \
#         -v yolo_free_v2_tiny \
#         -bs 16 \
#         --max_epoch 25 \
#         --wp_epoch 1 \
#         --eval_epoch 5 \
#         --ema \
#         --fp16 \
#         # --resume weights/coco/yolo_free_medium/yolo_free_medium_epoch_31_39.46.pth \
#         # --pretrained weights/coco/yolo_free_medium/yolo_free_medium_39.46.pth \
