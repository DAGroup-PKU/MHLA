# MHLA for Image Classification


## Requirements

- python >= 3.10
- torch >= 2.7
- timm >= 1.0
See `requirements.txt` for other dependencies.

To install the dependencies:
```bash
pip install -r requirements.txt
```


## Training

Train models using the `timm_train.py` script.

You can find a detailed training command in `scripts/train_mhla.sh`:

```bash
torchrun \
  --nproc_per_node=8 \
  --nnodes=1 \
  --master_port=12345 \
  timm_train.py \
  --model deit_small_pla_1d_v6 \
  --data-dir $data_dir_path \
  --opt adamw \
  --lr 1e-3 \
  --weight-decay 0.05 \
  --sched cosine \
  --epochs 200 \
  --warmup-epochs 10 \
  --warmup-lr 1e-6 \
  --batch-size 128 \
  --drop-path 0.1 \
  --reprob 0.25 \
  --mixup 0.2 \
  --cutmix 1.0 \
  --smoothing 0.1 \
  --aa rand-m9-mstd0.5-inc1 \
  --workers 32 \
  --model-ema \
  --model-ema-decay 0.9996 \
  --color-jitter 0.4 \
  --experiment pla_1d_v6 \
  --sched-on-updates \
  --log-wandb \
  --gp avg \
  --model-kwargs piece_size=4 transform=linear exp_sigma=1

```

Supported models registered in `models/timm_registers.py`.

The training script `timm_train.py` is modified from timm, and the configurable parameters are the same.



