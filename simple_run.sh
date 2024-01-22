#!/bin/bash
python model/run.py \
    --NUM_EPOCHS 30 \
    --TRAIN_FAST \
    --AMP \
    --SHARE_WEIGHTS \
    --ALIGN_FEATURES \
    --USE_MLP \
    --RUN_MODE train \
    --SAVE_CHECKPOINTS \
    --MODEL_NAME trial_run \
    
python model/run.py \
    --SHARE_WEIGHTS \
    --ALIGN_FEATURES \
    --USE_MLP \
    --RUN_MODE eval \
    --EVAL_GROUP cat1 \
    --LOAD_CHECKPOINT \
    --MODEL_NAME trial_run \
    --EVAL_SCORES_FNAME trial_scores