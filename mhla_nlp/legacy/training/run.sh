#!/bin/bash

exp_dir=exp/mhla-340M-10B

resume_arg=""
if compgen -G "$exp_dir/checkpoint-*" > /dev/null; then
  last_step=$(find "$exp_dir" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | \
    sed -n 's/^checkpoint-\([0-9]\+\)$/\1/p' | sort -nr | head -n1)
  if [ -n "$last_step" ]; then
    resume_arg="checkpoint=$exp_dir/checkpoint-$last_step"
  fi
fi


bash train.sh type=gla \
  lr=3e-4 \
  scheduler=cosine_with_min_lr \
  batch=16 \
  update=1 \
  warmup=1024 \
  steps=20480 \
  context=2048 \
  gpus=4 \
  nodes=1 \
  path=$exp_dir \
  project=fla \
  logging=4 \
  save=512 \
  limit=3 \
  model=configs/mhla_340M.json data=HuggingFaceFW/fineweb-edu name=sample-10BT cache=data/HuggingFaceFW/fineweb-edu/sample-10BT/train \
  $resume_arg
