#!/bin/bash

for VAL_FOLD in {1..9}; do
    for USE_MLP in mlp nomlp; do
        for SHARE_WEIGHTS in sharing nosharing; do
            for ALIGN_FEATURES in alignment noalignment; do
                python model/run.py \
                    --RUN_MODE train \
                    --TRAIN_FAST \
                    --AMP \
                    --VAL_FOLD $VAL_FOLD \
                    --TRAIN_LOGS \
                    --SAVE_CHECKPOINTS \
                    --MODEL_NAME "${USE_MLP}_${SHARE_WEIGHTS}_${ALIGN_FEATURES}_fold_${VAL_FOLD}" \
                    $(if [ "$ALIGN_FEATURES" == "alignment" ]; then echo "--ALIGN_FEATURES"; fi) \
                    $(if [ "$SHARE_WEIGHTS" == "sharing" ]; then echo "--ALIGN_FEATURES"; fi) \
                    $(if [ "$USE_MLP" == "mlp" ]; then echo "--USE_MLP"; fi) \

            done
        done
    done
done