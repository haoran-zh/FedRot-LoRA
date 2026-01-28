#!/bin/bash
# rolora, use device 2
python federatedscope/main.py --cfg federatedscope/glue/yamls/g_localstep20_rolora_B.yaml
python federatedscope/main.py --cfg federatedscope/glue/yamls/g_localstep20_rolora_B_lr5e-3.yaml
python federatedscope/main.py --cfg federatedscope/glue/yamls/global_rolora_B.yaml
python federatedscope/main.py --cfg federatedscope/glue/yamls/global_rolora_B_lr5e-3.yaml