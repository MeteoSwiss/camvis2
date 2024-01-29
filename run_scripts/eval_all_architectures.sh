# Copyright (c) 2024 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

#!/bin/bash

for VAL_FOLD in {1..9}; do
    COUNTER=0
    for USE_MLP in mlp nomlp; do
        for SHARE_WEIGHTS in sharing nosharing; do
            for ALIGN_FEATURES in alignment noalignment; do
                ((COUNTER ++))
                python model/run.py \
                    --RUN_MODE eval \
                    --VAL_FOLD $VAL_FOLD \
                    --LOAD_CHECKPOINT \
                    --MODEL_NAME "${USE_MLP}_${SHARE_WEIGHTS}_${ALIGN_FEATURES}_fold_${VAL_FOLD}" \
                    --EVAL_SCORES_FNAME all_architectures \
                    --EVAL_GROUP type_${COUNTER} \
                    $(if [ "$ALIGN_FEATURES" == "alignment" ]; then echo "--ALIGN_FEATURES"; fi) \
                    $(if [ "$SHARE_WEIGHTS" == "sharing" ]; then echo "--SHARE_WEIGHTS"; fi) \
                    $(if [ "$USE_MLP" == "mlp" ]; then echo "--USE_MLP"; fi) \
                
            done
        done
    done
done