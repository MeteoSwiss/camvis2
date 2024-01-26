# Copyright (c) 2024 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

#!/bin/bash

for VAL_FOLD in {1..9}; do
    for EVAL_GROUP in cat align; do
        python model/run.py \
            --RUN_MODE train \
            --TRAIN_FAST \
            --AMP \
            --SHARE_WEIGHTS \
            --USE_MLP \
            --VAL_FOLD $VAL_FOLD \
            --TRAIN_LOGS \
            --SAVE_CHECKPOINTS \
            --MODEL_NAME "${EVAL_GROUP}_features_fold_${VAL_FOLD}" \
            $(if [ "$EVAL_GROUP" == "align" ]; then echo "--ALIGN_FEATURES"; fi) \
            
    done
done