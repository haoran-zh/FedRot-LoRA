#!/bin/bash

LOCAL_STEPS=('lstep10' 'lstep20')
DATA='mnli@glue'
#LR_VALUES=('lr0.0005' 'lr0.001' 'lr0.002' 'lr0.005' 'lr0.01' 'lr0.02' 'lr0.05' 'lr0.1')
LR_VALUES=('lr0.005' 'lr0.02')
for LS in "${LOCAL_STEPS[@]}" # Loop through local steps
do
  for LR in "${LR_VALUES[@]}" # Nested loop through learning rates
  do
    python exp_results.py --lr "$LR" --lstep "$LS" --data "$DATA"
  done
done