# train YOLO with 8 GPUs
# 使用4张GPU来训练YOLO
python -m torch.distributed.run --nproc_per_node=4 train.py \
                                                    --cuda \
                                                    -dist \
                                                    -d coco \
                                                    --root /data/datasets/ \
                                                    -m yolovx_n \
                                                    -bs 64 \
                                                    -size 640 \
                                                    --wp_epoch 3 \
                                                    --max_epoch 300 \
                                                    --eval_epoch 10 \
                                                    --ema \
                                                    --fp16 \
                                                    --multi_scale \
