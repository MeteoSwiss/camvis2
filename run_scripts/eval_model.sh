# Copyright (c) 2024 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

#!/bin/bash

for VAL_FOLD in {1..9}; do
    python model/run.py \
        --RUN_MODE eval \
        --SHARE_WEIGHTS \
        --USE_MLP \
        --VAL_FOLD $VAL_FOLD \
        --LOAD_CHECKPOINT \
        --MODEL_NAME "mymodel_fold_${VAL_FOLD}" \
        --EVAL_SCORES_FNAME mymodel \

done