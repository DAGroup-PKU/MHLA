

np=1
config="configs/Wan_1300M_came8bit_fsdp_mhla_hybrid_2_3.yaml"
model_path="weights/mhla_videogen/model.safetensors"


torchrun --nproc_per_node=$np --standalone  \
        inference.py \
        --config_path=$config \
        --model_path=$model_path \
        --txt_file=samples_video.txt \
        --dataset=demo \
        --if_save_dirname true \
        --sample_nums=100



