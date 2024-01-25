#!/bin/bash

counter = 1
for VAL_FOLD in {1..9}; do
    for USE_MLP in mlp nomlp; do
        for SHARE_WEIGHTS in sharing nosharing; do
            for ALIGN_FEATURES in alignment noalignment; do
                python model/run.py \
                    --RUN_MODE eval \
                    --VAL_FOLD $VAL_FOLD \
                    --LOAD_CHECKPOINT \
                    --MODEL_NAME "${USE_MLP}_${SHARE_WEIGHTS}_${ALIGN_FEATURES}_fold_${VAL_FOLD}" \
                    --EVAL_SCORES_FNAME all_architectures \
                    --EVAL_GROUP type_${counter} \
                    $(if [ "$ALIGN_FEATURES" == "alignment" ]; then echo "--ALIGN_FEATURES"; fi) \
                    $(if [ "$SHARE_WEIGHTS" == "sharing" ]; then echo "--ALIGN_FEATURES"; fi) \
                    $(if [ "$USE_MLP" == "mlp" ]; then echo "--USE_MLP"; fi) \

                ((counter ++))
            done
        done
    done
done