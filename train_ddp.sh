# 8 GPUs
python -m torch.distributed.run --nproc_per_node=8 train.py \
                                                    --cuda \
                                                    -dist \
                                                    -d coco \
                                                    --root /data/datasets/ \
                                                    -m yolov1 \
                                                    -bs 128 \
                                                    -size 640 \
                                                    --wp_epoch 1 \
                                                    --max_epoch 150 \
                                                    --eval_epoch 10 \
                                                    --ema \
                                                    --fp16 \
                                                    --multi_scale \
