# Args parameters
MODEL=$1
DATASET=$2
DATASET_ROOT=$3
BATCH_SIZE=$4
WORLD_SIZE=$5
MASTER_PORT=$6
RESUME=$7

# MODEL setting
IMAGE_SIZE=640
FIND_UNUSED_PARAMS=False
if [[ $MODEL == *"yolov8"* ]]; then
    # Epoch setting
    MAX_EPOCH=500
    WP_EPOCH=3
    EVAL_EPOCH=10
    NO_AUG_EPOCH=20
elif [[ $MODEL == *"yolox"* ]]; then
    # Epoch setting
    MAX_EPOCH=300
    WP_EPOCH=3
    EVAL_EPOCH=10
    NO_AUG_EPOCH=20
elif [[ $MODEL == *"yolov7"* ]]; then
    # Epoch setting
    MAX_EPOCH=300
    WP_EPOCH=3
    EVAL_EPOCH=10
    NO_AUG_EPOCH=20
elif [[ $MODEL == *"yolov5"* ]]; then
    # Epoch setting
    MAX_EPOCH=300
    WP_EPOCH=3
    EVAL_EPOCH=10
    NO_AUG_EPOCH=20
elif [[ $MODEL == *"yolov4"* ]]; then
    # Epoch setting
    MAX_EPOCH=300
    WP_EPOCH=3
    EVAL_EPOCH=10
    NO_AUG_EPOCH=20
elif [[ $MODEL == *"yolov3"* ]]; then
    # Epoch setting
    MAX_EPOCH=300
    WP_EPOCH=3
    EVAL_EPOCH=10
    NO_AUG_EPOCH=20
else
    # Epoch setting
    MAX_EPOCH=150
    WP_EPOCH=3
    EVAL_EPOCH=10
    NO_AUG_EPOCH=10
fi

# -------------------------- Train Pipeline --------------------------
if [ $WORLD_SIZE == 1 ]; then
    python train.py \
            --cuda \
            --dataset ${DATASET} \
            --root ${DATASET_ROOT} \
            --model ${MODEL} \
            --batch_size ${BATCH_SIZE} \
            --img_size ${IMAGE_SIZE} \
            --wp_epoch ${WP_EPOCH} \
            --max_epoch ${MAX_EPOCH} \
            --eval_epoch ${EVAL_EPOCH} \
            --no_aug_epoch ${NO_AUG_EPOCH} \
            --resume ${RESUME} \
            --ema \
            --fp16 \
            --find_unused_parameters ${FIND_UNUSED_PARAMS} \
            --multi_scale
elif [[ $WORLD_SIZE -gt 1 && $WORLD_SIZE -le 8 ]]; then
    python -m torch.distributed.run --nproc_per_node=${WORLD_SIZE} --master_port ${MASTER_PORT} train.py \
            --cuda \
            -dist \
            --dataset ${DATASET} \
            --root ${DATASET_ROOT} \
            --model ${MODEL} \
            --batch_size ${BATCH_SIZE} \
            --img_size ${IMAGE_SIZE} \
            --wp_epoch ${WP_EPOCH} \
            --max_epoch ${MAX_EPOCH} \
            --eval_epoch ${EVAL_EPOCH} \
            --no_aug_epoch ${NO_AUG_EPOCH} \
            --resume ${RESUME} \
            --ema \
            --fp16 \
            --find_unused_parameters ${FIND_UNUSED_PARAMS} \
            --multi_scale \
            --sybn
else
    echo "The WORLD_SIZE is set to a value greater than 8, indicating the use of multi-machine \
          multi-card training mode, which is currently unsupported."
    exit 1
fi