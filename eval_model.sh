#!/bin/bash
python model/run.py \
    --RUN_MODE eval \
    --SHARE_WEIGHTS \
    --VAL_FOLD 1 \
    --LOAD_CHECKPOINT \
    --MODEL_NAME mymodel_fold_1 \
    --EVAL_SCORES_FNAME mymodel
    
python model/run.py \
    --RUN_MODE eval \
    --SHARE_WEIGHTS \
    --VAL_FOLD 2 \
    --LOAD_CHECKPOINT \
    --MODEL_NAME mymodel_fold_2 \
    --EVAL_SCORES_FNAME mymodel
    
python model/run.py \
    --RUN_MODE eval \
    --SHARE_WEIGHTS \
    --VAL_FOLD 3 \
    --LOAD_CHECKPOINT \
    --MODEL_NAME mymodel_fold_3 \
    --EVAL_SCORES_FNAME mymodel

python model/run.py \
    --RUN_MODE eval \
    --SHARE_WEIGHTS \
    --VAL_FOLD 4 \
    --LOAD_CHECKPOINT \
    --MODEL_NAME mymodel_fold_4 \
    --EVAL_SCORES_FNAME mymodel

python model/run.py \
    --RUN_MODE eval \
    --SHARE_WEIGHTS \
    --VAL_FOLD 5 \
    --LOAD_CHECKPOINT \
    --MODEL_NAME mymodel_fold_5 \
    --EVAL_SCORES_FNAME mymodel

python model/run.py \
    --RUN_MODE eval \
    --SHARE_WEIGHTS \
    --VAL_FOLD 6 \
    --LOAD_CHECKPOINT \
    --MODEL_NAME mymodel_fold_6 \
    --EVAL_SCORES_FNAME mymodel

python model/run.py \
    --RUN_MODE eval \
    --SHARE_WEIGHTS \
    --VAL_FOLD 7 \
    --LOAD_CHECKPOINT \
    --MODEL_NAME mymodel_fold_7 \
    --EVAL_SCORES_FNAME mymodel
    
python model/run.py \
    --RUN_MODE eval \
    --SHARE_WEIGHTS \
    --VAL_FOLD 8 \
    --LOAD_CHECKPOINT \
    --MODEL_NAME mymodel_fold_8 \
    --EVAL_SCORES_FNAME mymodel
    
python model/run.py \
    --RUN_MODE eval \
    --SHARE_WEIGHTS \
    --VAL_FOLD 9 \
    --LOAD_CHECKPOINT \
    --MODEL_NAME mymodel_fold_9 \
    --EVAL_SCORES_FNAME mymodel