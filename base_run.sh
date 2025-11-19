#!/bin/bash
BASE_YAML="federatedscope/glue/yamls/base_worotation.yaml"
SEEDS=(11 12 13)
DEVICE_ID=2
TOTAL_TRAIN=5000
LOCAL_STEPS=(30 40 50)
DATA='mnli@glue'
LR_VALUES=(2e-2 5e-3)

echo "Base Config: $BASE_YAML"
for LS in "${LOCAL_STEPS[@]}"
do
    # Calculate TOTAL_ROUNDS based on current LOCAL_STEPS
    # This uses integer arithmetic
    ROUNDS=$((TOTAL_TRAIN / LS))

    for LR in "${LR_VALUES[@]}"
    do
        for SEED in "${SEEDS[@]}"
        do
            python federatedscope/main.py \
                --cfg $BASE_YAML \
                device $DEVICE_ID \
                seed $SEED \
                federate.total_round_num $ROUNDS \
                data.type $DATA \
                train.local_update_steps $LS \
                train.optimizer.lr $LR
        done
    done
done