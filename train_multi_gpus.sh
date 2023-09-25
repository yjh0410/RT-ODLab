# -------------------------- Train RTCDet --------------------------
python -m torch.distributed.run --nproc_per_node=8 train.py \
                                                    --cuda \
                                                    -dist \
                                                    -d coco \
                                                    --root /data/datasets/ \
                                                    -m rtcdet_t \
                                                    -bs 128 \
                                                    -size 640 \
                                                    --wp_epoch 3 \
                                                    --max_epoch 300 \
                                                    --eval_epoch 10 \
                                                    --no_aug_epoch 20 \
                                                    --ema \
                                                    --fp16 \
                                                    --sybn \
                                                    --multi_scale \
                                                    --save_folder weights/ \
                                                    # --load_cache \
                                                    # --resume weights/coco/yolox_l/yolox_l_best.pth \

# -------------------------- Train YOLOX & YOLOv7 --------------------------
# python -m torch.distributed.run --nproc_per_node=8 train.py \
#                                                     --cuda \
#                                                     -dist \
#                                                     -d coco \
#                                                     --root /data/datasets/ \
#                                                     -m rtcdet_n \
#                                                     -bs 64 \
#                                                     -size 640 \
#                                                     --wp_epoch 3 \
#                                                     --max_epoch 300 \
#                                                     --eval_epoch 10 \
#                                                     --no_aug_epoch 15 \
#                                                     --ema \
#                                                     --fp16 \
#                                                     --sybn \
#                                                     --multi_scale \
#                                                     # --load_cache \
#                                                     # --resume weights/coco/yolox_l/yolox_l_best.pth \

# -------------------------- Train YOLOv1~v5 --------------------------
# python -m torch.distributed.run --nproc_per_node=8 train.py \
#                                                     --cuda \
#                                                     -dist \
#                                                     -d coco \
#                                                     --root /data/datasets/ \
#                                                     -m yolov5_l\
#                                                     -bs 128 \
#                                                     -size 640 \
#                                                     --wp_epoch 3 \
#                                                     --max_epoch 300 \
#                                                     --eval_epoch 10 \
#                                                     --no_aug_epoch 10 \
#                                                     --ema \
#                                                     --fp16 \
#                                                     --sybn \
#                                                     --multi_scale \
#                                                     # --load_cache
#                                                     # --resume weights/coco/yolov5_l/yolov5_l_best.pth \
