# FedRot-LoRA: Mitigating Rotational Misalignment in Federated LoRA

## Implementation of FedRot-LoRA

The core FedRot-LoRA logic is implemented in the following files:
- `federatedscope/core/workers/client.py`
- `federatedscope/core/workers/server.py`
- `federatedscope/core/configs/cfg_llm.py`
- `federatedscope/rotation_alignment_tools.py`

## Installation

Our code is based on Python version 3.10 and PyTorch version 2.1.0. 
You can install all the dependencies with the following command:
```shell
conda create -n fedrot-lora python=3.10
conda activate fedrot-lora
# conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia  # this command may have problems.
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -e .[llm]
pip install evaluate
```

## Training

Now, we can fine-tune LLMs with FedRot-LoRA:

```shell
python federatedscope/main.py --cfg federatedscope/glue/yamls/base_fedrot_lora.yaml
```
```shell
python federatedscope/main.py --cfg federatedscope/llm/yamls/base_fedrot_lora.yaml
```

## Acknowledgement

Our implementation is built on top of the FedSA-LoRA codebase: https://github.com/Pengxin-Guo/FedSA-LoRA

We would like to thank the authors for releasing the public repository: [FederatedScope-LLM](https://github.com/alibaba/FederatedScope/tree/llm).
