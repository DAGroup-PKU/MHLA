# MHLA for DiT



## Requirements

- python >= 3.10
- torch >= 2.7
- torchvision
See `requirements.txt` for other dependencies.

## Training
```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    train.py \
    --model DiT-S/2 --image-size 256  \
    --global-batch-size 256 --epochs 80 --ckpt-every 1 \
    --feature-path \"${FEATURE_PATH}\" \
    --results-dir \"${RESULTS_DIR}\" \
    --experiment "DiT-S/2" \
    --model-kwargs block_size=16
```
`block_size` indicates the size of "token-level head". The value of block_size must be a perfect square of some integer and must evenly divide the embedding length. Common values are 4, 16, 64, and so on.

