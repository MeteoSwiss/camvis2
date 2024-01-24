#!/bin/bash
python model/run.py \
    --RUN_MODE train \
    --TRAIN_FAST \
    --AMP \
    --SHARE_WEIGHTS \
    --VAL_FOLD 1 \
    --TRAIN_LOGS \
    --SAVE_CHECKPOINTS \
    --MODEL_NAME mymodel_fold_1 \

python model/run.py \
    --RUN_MODE train \
    --TRAIN_FAST \
    --AMP \
    --SHARE_WEIGHTS \
    --VAL_FOLD 2 \
    --TRAIN_LOGS \
    --SAVE_CHECKPOINTS \
    --MODEL_NAME mymodel_fold_2 \

python model/run.py \
    --RUN_MODE train \
    --TRAIN_FAST \
    --AMP \
    --SHARE_WEIGHTS \
    --VAL_FOLD 3 \
    --TRAIN_LOGS \
    --SAVE_CHECKPOINTS \
    --MODEL_NAME mymodel_fold_3 \

python model/run.py \
    --RUN_MODE train \
    --TRAIN_FAST \
    --AMP \
    --SHARE_WEIGHTS \
    --VAL_FOLD 4 \
    --TRAIN_LOGS \
    --SAVE_CHECKPOINTS \
    --MODEL_NAME mymodel_fold_4 \

python model/run.py \
    --RUN_MODE train \
    --TRAIN_FAST \
    --AMP \
    --SHARE_WEIGHTS \
    --VAL_FOLD 5 \
    --TRAIN_LOGS \
    --SAVE_CHECKPOINTS \
    --MODEL_NAME mymodel_fold_5 \

python model/run.py \
    --RUN_MODE train \
    --TRAIN_FAST \
    --AMP \
    --SHARE_WEIGHTS \
    --VAL_FOLD 6 \
    --TRAIN_LOGS \
    --SAVE_CHECKPOINTS \
    --MODEL_NAME mymodel_fold_6 \

python model/run.py \
    --RUN_MODE train \
    --TRAIN_FAST \
    --AMP \
    --SHARE_WEIGHTS \
    --VAL_FOLD 7 \
    --TRAIN_LOGS \
    --SAVE_CHECKPOINTS \
    --MODEL_NAME mymodel_fold_7 \

python model/run.py \
    --RUN_MODE train \
    --TRAIN_FAST \
    --AMP \
    --SHARE_WEIGHTS \
    --VAL_FOLD 8 \
    --TRAIN_LOGS \
    --SAVE_CHECKPOINTS \
    --MODEL_NAME mymodel_fold_8 \

python model/run.py \
    --RUN_MODE train \
    --TRAIN_FAST \
    --AMP \
    --SHARE_WEIGHTS \
    --VAL_FOLD 9 \
    --TRAIN_LOGS \
    --SAVE_CHECKPOINTS \
    --MODEL_NAME mymodel_fold_9 \