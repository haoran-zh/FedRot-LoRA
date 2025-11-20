#!/bin/bash
BASE_YAMLS=(
    "federatedscope/glue/yamls/base_fedlora2.yaml"
    "federatedscope/glue/yamls/base_rolora.yaml"
        "federatedscope/glue/yamls/base_worotation.yaml"
)
SEEDS=(12)  # let's only do 1 random seed for quick testing
DEVICE_ID=2
TOTAL_TRAIN=5000
LOCAL_STEPS=(10 20)
DATA='rte@glue'
LR_VALUES=(5e-4 1e-3 2e-3 5e-3 1e-2 2e-2 5e-2 1e-1)

for BASE_YAML in "${BASE_YAMLS[@]}"
do
    echo "--- Base Config: $BASE_YAML ---"

    # Loop over local steps
    for LS in "${LOCAL_STEPS[@]}"
    do
        # Calculate TOTAL_ROUNDS based on current LOCAL_STEPS
        # This uses integer arithmetic
        ROUNDS=$((TOTAL_TRAIN / LS))
        echo "Calculated total rounds: $ROUNDS for local steps: $LS"

        # Loop over learning rate values
        for LR in "${LR_VALUES[@]}"
        do
            # Loop over seeds
            for SEED in "${SEEDS[@]}"
            do
                echo "Running: BASE=$BASE_YAML, LS=$LS, LR=$LR, SEED=$SEED, ROUNDS=$ROUNDS"

                # Execute the Python script with all arguments
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
done