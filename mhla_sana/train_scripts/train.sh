#!/bin/bash

 TRITON_PRINT_AUTOTUNING=1 torchrun \
     --nproc_per_node=8 \
     train_scripts/train.py \
     --config_path=configs/sana_config/512ms/Sana_600M_img512_MHLA.yaml \
     --work_dir=output/debug_512 \
     --name=sana_600M_512px_MHLA \
     --resume_from=latest \
     --train.use_fsdp=false  \
     --debug=true