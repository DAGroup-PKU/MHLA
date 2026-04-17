# MHLA for Sana

This directory is adapted from [Sana](https://github.com/NVlabs/Sana) (NVIDIA) and trimmed for the MHLA paper experiments. We keep a layout compatible with the upstream project so you can align it with the full [Sana](https://github.com/NVlabs/Sana) tree if needed. Using only the code here is enough to reproduce the Sana + MHLA image-generation setup described in the paper.

## Requirements

- Python >= 3.10
- PyTorch with a CUDA build matching your GPU stack (see `[tool.pip]` in `pyproject.toml` for the suggested PyTorch wheel index)
- Core libraries are listed in `pyproject.toml` (e.g. `diffusers`, `transformers`, `accelerate`, `xformers`, `triton`, `einops`, `timm`, `webdataset`, and others)

To install the package in editable mode (recommended):

```bash
pip install -U pip
pip install -e .
```

Optional: you can use `environment_setup.sh` as a starting point for a Conda environment; adjust CUDA / `xformers` versions to match your machine and the pins in `pyproject.toml`.

## Data preparation

Point `data.data_dir` in your training YAML (for example `configs/sana_config/512ms/Sana_600M_img512_MHLA.yaml`) to your image–caption dataset. The sample config uses `asset/example_data` as a placeholder; replace it with your own WebDataset or Sana-style data layout before large-scale training.

## Training

Launch distributed training with `torchrun`, following `train_scripts/train.sh`. The default script uses the MHLA config and 8 GPUs:

```bash
bash train_scripts/train.sh
```

which runs `train_scripts/train.py` with `configs/sana_config/512ms/Sana_600M_img512_MHLA.yaml`. Override `--work_dir`, `--name`, data paths, and other flags as needed for your cluster.

## Inference

Single-image inference loads the MHLA checkpoint from Hugging Face (see `infer_single.py`):

```bash
python infer_single.py
```

Adjust the YAML path, `from_pretrained` URI, prompt, and resolution in the script to match your checkpoint and use case.

Pretrained weights for this variant are published under the MHLA Hugging Face collection; see the root [README.md](../README.md) for links.
