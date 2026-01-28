#!/bin/bash
# share_AB without rotation, use device 1
python federatedscope/main.py --cfg federatedscope/glue/yamls/g_localstep20_fedlora2_shareAB_wo_rotation.yaml
python federatedscope/main.py --cfg federatedscope/glue/yamls/g_localstep20_fedlora2_shareAB_wo_rotation_lr5e-3.yaml
python federatedscope/main.py --cfg federatedscope/glue/yamls/global_fedlora2_shareAB_wo_rotation.yaml
python federatedscope/main.py --cfg federatedscope/glue/yamls/global_fedlora2_shareAB_wo_rotation_lr5e-3.yaml
