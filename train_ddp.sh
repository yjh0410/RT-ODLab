# train YOLO with 8 GPUs
# 使用 8 GPU来训练YOLO
python -m torch.distributed.run --nproc_per_node=8 train.py \
                                                    --cuda \
                                                    -dist \
                                                    -d coco \
                                                    --root /data/datasets/ \
                                                    -m rtcdet_v1_x\
                                                    -bs 96 \
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
