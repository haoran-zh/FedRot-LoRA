#!/bin/bash
# use device 0
python federatedscope/main.py --cfg federatedscope/glue/yamls/g_localstep20_fedlora2_shareAB.yaml
python federatedscope/main.py --cfg federatedscope/glue/yamls/g_localstep20_fedlora2_shareAB_lr5e-3.yaml
python federatedscope/main.py --cfg federatedscope/glue/yamls/global_fedlora2_shareAB.yaml
python federatedscope/main.py --cfg federatedscope/glue/yamls/global_fedlora2_shareAB_lr5e-3.yaml
