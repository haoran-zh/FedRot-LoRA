#!/bin/bash

LOCAL_STEPS=('lstep10' 'lstep20' 'lstep30' 'lstep40' 'lstep50')
DATA='mnli@glue'
LR_VALUES=('lr0.02' 'lr0.005')

for LS in "${LOCAL_STEPS[@]}" # Loop through local steps
do
  for LR in "${LR_VALUES[@]}" # Nested loop through learning rates
  do
    python exp_results.py --lr "$LR" --lstep "$LS" --data "$DATA"
  done
done