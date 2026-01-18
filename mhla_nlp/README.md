# MHLA for NLP

This repository is modified and trimmed down from [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention). We have maintained the file structure of the original repository and removed redundant files. You can place the files corresponding to this sub-project into [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) to obtain the complete content of the original repository. Of course, using only the content in this repository is also sufficient to reproduce the MHLA results presented in the paper.

## Requirements

- python >= 3.10
- torch >= 2.5
- triton >= 3.0
- einops
- transformers >=4.45.0
- datasets >=3.3.0
See `requirements.txt` for other dependencies.

To install the dependencies:
```bash
pip install -r requirements.txt
```
  

## Data Preparing

Follow [legacy/training/README.md](https://www.google.com/search?q=legacy/training/README.md). 

## Training

Follow [legacy/training/README.md](https://www.google.com/search?q=legacy/training/README.md). We directly replaced the attention mechanism with MHLA in the GLAModel and trained it for a fair comparison. The command for training MHLA is as follows:

```bash
bash train.sh type=gla \
  lr=3e-4 \
  scheduler=cosine_with_min_lr \
  batch=16 \
  update=1 \
  warmup=1024 \
  steps=20480 \
  context=2048 \
  gpus=8 \
  nodes=1 \
  path=exp/MHLA-340M-10B \
  project=mhla \
  logging=4 \
  save=512 \
  limit=3 \
  model=configs/mhla_340M.json data=HuggingFaceFW/fineweb-edu name=sample-10BT cache=data/HuggingFaceFW/fineweb-edu/sample-10BT/train

```
