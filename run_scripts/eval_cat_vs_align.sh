#!/bin/bash

for VAL_FOLD in {1..9}; do
    for EVAL_GROUP in cat align; do
        python model/run.py \
            --RUN_MODE eval \
            --SHARE_WEIGHTS \
            --USE_MLP \
            --VAL_FOLD $VAL_FOLD \
            --LOAD_CHECKPOINT \
            --MODEL_NAME "${EVAL_GROUP}_features_fold_${VAL_FOLD}" \
            --EVAL_SCORES_FNAME cat_vs_align \
            --EVAL_GROUP $EVAL_GROUP \
            $(if [ "$EVAL_GROUP" == "align" ]; then echo "--ALIGN_FEATURES"; fi) \

    done
done