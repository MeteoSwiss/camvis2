#!/bin/bash

for VAL_FOLD in {1..9}; do
    python model/run.py \
        --RUN_MODE train \
        --TRAIN_FAST \
        --AMP \
        --SHARE_WEIGHTS \
        --VAL_FOLD $VAL_FOLD \
        --TRAIN_LOGS \
        --SAVE_CHECKPOINTS \
        --MODEL_NAME "mymodel_fold_${VAL_FOLD}" \

done