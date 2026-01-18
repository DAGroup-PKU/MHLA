#!/bin/bash
set -e

work_dir=output/mhla
np=8


if [[ $1 == *.yaml ]]; then
    config=$1
    shift
else
    config="configs/Wan_1300M_came8bit_fsdp_mhla.yaml"
    echo "Only support .yaml files, but get $1. Set to --config_path=$config"
fi


cmd="TRITON_PRINT_AUTOTUNING=1 \
    torchrun --nproc_per_node=$np --master_port=$((RANDOM % 10000 + 20000))  \
        train_wan.py \
        --config_path=$config \
        --work_dir=$work_dir \
        --train.log_interval=1 \
        --name=tmp \
        --resume_from=latest \
        $@"

echo $cmd
eval $cmd
