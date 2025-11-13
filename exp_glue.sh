#!/bin/bash
python federatedscope/main.py --cfg federatedscope/glue/yamls/global_rolora.yaml
python federatedscope/main.py --cfg federatedscope/glue/yamls/global_fedlora2_shareAB.yaml
python federatedscope/main.py --cfg federatedscope/glue/yamls/global_fedlora2_shareAB_wo_rotation.yaml
python federatedscope/main.py --cfg federatedscope/glue/yamls/global_fedlora2_swap.yaml
