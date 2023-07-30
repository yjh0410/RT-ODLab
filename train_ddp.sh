# train YOLO with 4 GPUs
# 使用 4 GPU来训练YOLO
python -m torch.distributed.run --nproc_per_node=8 train.py \
                                                    --cuda \
                                                    -dist \
                                                    -d coco \
                                                    --root /data/datasets/ \
                                                    -m yolox_l\
                                                    -bs 64 \
                                                    -size 640 \
                                                    --wp_epoch 3 \
                                                    --max_epoch 300 \
                                                    --eval_epoch 10 \
                                                    --no_aug_epoch 20 \
                                                    --ema \
                                                    --fp16 \
                                                    --sybn \
                                                    --multi_scale \
                                                    # --resume weights/coco/yolovx_s/yolovx_s_best.pth \
