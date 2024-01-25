#!/bin/bash

for VAL_FOLD in {1..9}; do
    python model/run.py \
        --RUN_MODE eval \
        --SHARE_WEIGHTS \
        --VAL_FOLD $VAL_FOLD \
        --LOAD_CHECKPOINT \
        --MODEL_NAME "mymodel_fold_${VAL_FOLD}" \
        --EVAL_SCORES_FNAME mymodel \

done