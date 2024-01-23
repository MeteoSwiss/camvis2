#!/bin/bash
python model/run.py \
    --RUN_MODE train \
    --TRAIN_FAST \
    --AMP \
    --SHARE_WEIGHTS \
    --USE_MLP \
    --VAL_FOLD 1 \
    --EVAL_GROUP cat \
    --TRAIN_LOGS \
    --SAVE_CHECKPOINTS \
    --MODEL_NAME cat_features_fold_1 \

python model/run.py \
    --RUN_MODE train \
    --TRAIN_FAST \
    --AMP \
    --ALIGN_FEATURES \
    --SHARE_WEIGHTS \
    --USE_MLP \
    --VAL_FOLD 1 \
    --EVAL_GROUP align \
    --TRAIN_LOGS \
    --SAVE_CHECKPOINTS \
    --MODEL_NAME align_features_fold_1 \

python model/run.py \
    --RUN_MODE train \
    --TRAIN_FAST \
    --AMP \
    --SHARE_WEIGHTS \
    --USE_MLP \
    --VAL_FOLD 2 \
    --EVAL_GROUP cat \
    --TRAIN_LOGS \
    --SAVE_CHECKPOINTS \
    --MODEL_NAME cat_features_fold_2 \

python model/run.py \
    --RUN_MODE train \
    --TRAIN_FAST \
    --AMP \
    --ALIGN_FEATURES \
    --SHARE_WEIGHTS \
    --USE_MLP \
    --VAL_FOLD 2 \
    --EVAL_GROUP align \
    --TRAIN_LOGS \
    --SAVE_CHECKPOINTS \
    --MODEL_NAME align_features_fold_2 \

python model/run.py \
    --RUN_MODE train \
    --TRAIN_FAST \
    --AMP \
    --SHARE_WEIGHTS \
    --USE_MLP \
    --VAL_FOLD 3 \
    --EVAL_GROUP cat \
    --TRAIN_LOGS \
    --SAVE_CHECKPOINTS \
    --MODEL_NAME cat_features_fold_3 \

python model/run.py \
    --RUN_MODE train \
    --TRAIN_FAST \
    --AMP \
    --ALIGN_FEATURES \
    --SHARE_WEIGHTS \
    --USE_MLP \
    --VAL_FOLD 3 \
    --EVAL_GROUP align \
    --TRAIN_LOGS \
    --SAVE_CHECKPOINTS \
    --MODEL_NAME align_features_fold_3 \

python model/run.py \
    --RUN_MODE train \
    --TRAIN_FAST \
    --AMP \
    --SHARE_WEIGHTS \
    --USE_MLP \
    --VAL_FOLD 4 \
    --EVAL_GROUP cat \
    --TRAIN_LOGS \
    --SAVE_CHECKPOINTS \
    --MODEL_NAME cat_features_fold_4 \

python model/run.py \
    --RUN_MODE train \
    --TRAIN_FAST \
    --AMP \
    --ALIGN_FEATURES \
    --SHARE_WEIGHTS \
    --USE_MLP \
    --VAL_FOLD 4 \
    --EVAL_GROUP align \
    --TRAIN_LOGS \
    --SAVE_CHECKPOINTS \
    --MODEL_NAME align_features_fold_4 \

python model/run.py \
    --RUN_MODE train \
    --TRAIN_FAST \
    --AMP \
    --SHARE_WEIGHTS \
    --USE_MLP \
    --VAL_FOLD 5 \
    --EVAL_GROUP cat \
    --TRAIN_LOGS \
    --SAVE_CHECKPOINTS \
    --MODEL_NAME cat_features_fold_5 \

python model/run.py \
    --RUN_MODE train \
    --TRAIN_FAST \
    --AMP \
    --ALIGN_FEATURES \
    --SHARE_WEIGHTS \
    --USE_MLP \
    --VAL_FOLD 5 \
    --EVAL_GROUP align \
    --TRAIN_LOGS \
    --SAVE_CHECKPOINTS \
    --MODEL_NAME align_features_fold_5 \

python model/run.py \
    --RUN_MODE train \
    --TRAIN_FAST \
    --AMP \
    --SHARE_WEIGHTS \
    --USE_MLP \
    --VAL_FOLD 6 \
    --EVAL_GROUP cat \
    --TRAIN_LOGS \
    --SAVE_CHECKPOINTS \
    --MODEL_NAME cat_features_fold_6 \

python model/run.py \
    --RUN_MODE train \
    --TRAIN_FAST \
    --AMP \
    --ALIGN_FEATURES \
    --SHARE_WEIGHTS \
    --USE_MLP \
    --VAL_FOLD 6 \
    --EVAL_GROUP align \
    --TRAIN_LOGS \
    --SAVE_CHECKPOINTS \
    --MODEL_NAME align_features_fold_6 \

python model/run.py \
    --RUN_MODE train \
    --TRAIN_FAST \
    --AMP \
    --SHARE_WEIGHTS \
    --USE_MLP \
    --VAL_FOLD 7 \
    --EVAL_GROUP cat \
    --TRAIN_LOGS \
    --SAVE_CHECKPOINTS \
    --MODEL_NAME cat_features_fold_7 \

python model/run.py \
    --RUN_MODE train \
    --TRAIN_FAST \
    --AMP \
    --ALIGN_FEATURES \
    --SHARE_WEIGHTS \
    --USE_MLP \
    --VAL_FOLD 7 \
    --EVAL_GROUP align \
    --TRAIN_LOGS \
    --SAVE_CHECKPOINTS \
    --MODEL_NAME align_features_fold_7 \

python model/run.py \
    --RUN_MODE train \
    --TRAIN_FAST \
    --AMP \
    --SHARE_WEIGHTS \
    --USE_MLP \
    --VAL_FOLD 8 \
    --EVAL_GROUP cat \
    --TRAIN_LOGS \
    --SAVE_CHECKPOINTS \
    --MODEL_NAME cat_features_fold_8 \

python model/run.py \
    --RUN_MODE train \
    --TRAIN_FAST \
    --AMP \
    --ALIGN_FEATURES \
    --SHARE_WEIGHTS \
    --USE_MLP \
    --VAL_FOLD 8 \
    --EVAL_GROUP align \
    --TRAIN_LOGS \
    --SAVE_CHECKPOINTS \
    --MODEL_NAME align_features_fold_8 \

python model/run.py \
    --RUN_MODE train \
    --TRAIN_FAST \
    --AMP \
    --SHARE_WEIGHTS \
    --USE_MLP \
    --VAL_FOLD 9 \
    --EVAL_GROUP cat \
    --TRAIN_LOGS \
    --SAVE_CHECKPOINTS \
    --MODEL_NAME cat_features_fold_9 \

python model/run.py \
    --RUN_MODE train \
    --TRAIN_FAST \
    --AMP \
    --ALIGN_FEATURES \
    --SHARE_WEIGHTS \
    --USE_MLP \
    --VAL_FOLD 9 \
    --EVAL_GROUP align \
    --TRAIN_LOGS \
    --SAVE_CHECKPOINTS \
    --MODEL_NAME align_features_fold_9 \