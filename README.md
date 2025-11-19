# FedLoRA2

The implementation of **F**ederated **Lo**w-**R**ank **A**daption via **R**otation **A**lignment (FedLoRA2). 

The code is based on the implementation of [Selective Aggregation for Low-Rank Adaptation in Federated Learning](https://openreview.net/forum?id=iX3uESGdsO) [ICLR 2025]. \
[Pengxin Guo](https://pengxin-guo.github.io), [Shuang Zeng](https://scholar.google.com/citations?user=yTP1oqkAAAAJ&hl=en), Yanran Wang, Huijie Fan, Feifei Wang, and [Liangqiong Qu](https://liangqiong.github.io).

## Installation

Our code is based on Python version 3.10 and PyTorch version 2.1.0. 
You can install all the dependencies with the following command:
```shell
conda create -n fedsa-lora python=3.10
conda activate fedsa-lora
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -e .[llm]
```

The following command may bring the error "undefined symbol: iJIT_IsProfilingActive"
```shell
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Replace it with
```shell
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```


## Training

Now, we can fine-tune a LLM with FedSA-LoRA:

```shell
python federatedscope/main.py --cfg federatedscope/glue/yamls/fedsa-lora.yaml
```

To use Llama-3-8B, login huggingface first:
```shell
huggingface-cli login
```
Then:
```shell
python federatedscope/main.py --cfg federatedscope/llm/yamls/fedsa-lora.yaml
```

## Acknowledgement

We would like to thank the authors for releasing the public repository: [FederatedScope-LLM](https://github.com/alibaba/FederatedScope/tree/llm).
