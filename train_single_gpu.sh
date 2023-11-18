# Dataset setting
DATASET="coco"
DATA_ROOT="/data/datasets/"
# DATA_ROOT="/Users/liuhaoran/Desktop/python_work/object-detection/dataset/"

# MODEL setting
MODEL="yolov8_n"
IMAGE_SIZE=640
RESUME="None"
if [[ $MODEL == *"yolov8"* ]]; then
    # Epoch setting
    MAX_EPOCH=500
    BATCH_SIZE=16
    WP_EPOCH=3
    EVAL_EPOCH=10
    NO_AUG_EPOCH=20
elif [[ $MODEL == *"yolox"* ]]; then
    # Epoch setting
    MAX_EPOCH=300
    BATCH_SIZE=16
    WP_EPOCH=3
    EVAL_EPOCH=10
    NO_AUG_EPOCH=15
elif [[ $MODEL == *"yolov7"* ]]; then
    # Epoch setting
    MAX_EPOCH=300
    BATCH_SIZE=16
    WP_EPOCH=3
    EVAL_EPOCH=10
    NO_AUG_EPOCH=20
elif [[ $MODEL == *"yolov5"* || $MODEL == *"yolov4"* || $MODEL == *"yolov3"* ]]; then
    # Epoch setting
    MAX_EPOCH=300
    BATCH_SIZE=16
    WP_EPOCH=3
    EVAL_EPOCH=10
    NO_AUG_EPOCH=15
else
    # Epoch setting
    MAX_EPOCH=150
    BATCH_SIZE=16
    WP_EPOCH=3
    EVAL_EPOCH=10
    NO_AUG_EPOCH=0
fi

# -------------------------- Train Pipeline --------------------------
python train.py \
        --cuda \
        --dataset ${DATASET} \
        --root ${DATA_ROOT} \
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
        --multi_scale
