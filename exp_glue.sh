#!/bin/bash
# use device 0
BASE_YAML="federatedscope/glue/yamls/g_localstep20_fedlora2_shareAB.yaml"

# --- Experiment 1: Iterate over Random Seeds ---
echo "Running experiments with different seeds..."
for SEED in 11 12 13
do
    echo "Starting run with seed: $SEED"
    python federatedscope/main.py \
        --cfg $BASE_YAML \
        seed $SEED
done
