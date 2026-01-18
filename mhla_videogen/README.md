# MHLA for VideoGen

This repository is modified and trimmed down from [Sana](https://github.com/NVlabs/Sana). The training data format is consistent with Sana. Please refer to its repository for data preparation.

## Requirements

- python >= 3.10
- torch >= 2.7.0
- torchvision
- diffusers
- transformers
- accelerate
- peft
See `requirements.txt` for other dependencies.

To install the dependencies:
```bash
pip install -r requirements.txt
```


## Training

See `train_wan.sh`. You can train the models with `train_wan.py`.

```bash
torchrun --nproc_per_node=$np --master_port=$((RANDOM % 10000 + 20000))  \
        train_wan.py \
        --config_path=$config \
        --work_dir=$work_dir \
        --train.log_interval=1 \
        --name=test \
        --resume_from=latest 
```


## Inference

You can use the trained model to generate videos through `inference.py` as follows:


```bash
torchrun --nproc_per_node=$np --master_port=$((RANDOM % 10000 + 20000))  \
        inference.py \
        --config_path=$config \
        --model_path=$model_path \
        --txt_file=$txt_file 

```
you should put the prompts in `$txt_file`. You can download the model checkpoint from [huggingface](https://huggingface.co/DAGroup-PKU/MHLA)






