# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import math
import os
import re
import subprocess
import tarfile
import time
import warnings
from dataclasses import dataclass, field
from functools import partial

# from datetime import datetime
from typing import List, Optional

import imageio
import numpy as np
import pyrallis
import torch
from accelerate import Accelerator, InitProcessGroupKwargs, skip_first_batches
from peft import LoraConfig, get_peft_model
from termcolor import colored
from torchvision.utils import save_image
from tqdm import tqdm

warnings.filterwarnings("ignore")  # ignore warning

import datetime

from diffusion import DPMS, FlowEuler, SASolverSampler, UniPC
from diffusion.data.datasets.utils import (
    ASPECT_RATIO_512_TEST,
    ASPECT_RATIO_1024_TEST,
    ASPECT_RATIO_2048_TEST,
    ASPECT_RATIO_4096_TEST,
)
from diffusion.data.transforms import read_image_from_path
from diffusion.model.builder import (
    build_model,
    encode_image,
    get_image_encoder,
    get_tokenizer_and_text_encoder,
    get_vae,
    vae_decode,
    vae_encode,
)
from diffusion.model.utils import get_weight_dtype, prepare_prompt_ar
from diffusion.model.wan import T5EncoderModel, WanLinearAttentionModel, WanModel, WanVAE, init_model_configs
from diffusion.model.wan.fsdp_utils import shard_model
from diffusion.utils.config import SanaConfig, model_init_config
from diffusion.utils.config_wan import WanConfig
from diffusion.utils.dist_utils import flush, get_world_size
from diffusion.utils.logger import get_root_logger
from tools.download import find_model
# from tools.prompt_extender.qwen_prompt_extender import QwenPromptExpander


def set_fsdp_env():
    # Basic FSDP settings
    os.environ["ACCELERATE_USE_FSDP"] = "true"

    # Auto wrapping policy
    os.environ["FSDP_AUTO_WRAP_POLICY"] = "TRANSFORMER_BASED_WRAP"
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = "WanAttentionBlock"  # Your transformer block name
    # Performance optimization settings
    os.environ["FSDP_BACKWARD_PREFETCH"] = "BACKWARD_PRE"
    os.environ["FSDP_FORWARD_PREFETCH"] = "false"

    # State dict settings
    os.environ["FSDP_STATE_DICT_TYPE"] = "FULL_STATE_DICT"
    os.environ["FSDP_SYNC_MODULE_STATES"] = "true"
    os.environ["FSDP_USE_ORIG_PARAMS"] = "true"

    # Sharding strategy
    # [1] FULL_SHARD (shards optimizer states, gradients and parameters), [2] SHARD_GRAD_OP (shards optimizer states and gradients), [3] NO_SHARD (DDP), [4] HYBRID_SHARD (shards optimizer states, gradients and parameters within each node while each node has full copy), [5] HYBRID_SHARD_ZERO2 (shards optimizer states and gradients within each node while each node has full copy).
    os.environ["FSDP_SHARDING_STRATEGY"] = "FULL_SHARD"

    # Memory optimization settings (optional)
    os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "true"  # "false"
    os.environ["FSDP_OFFLOAD_PARAMS"] = "false"  # "false"

    # Precision settings
    os.environ["FSDP_REDUCE_SCATTER_PRECISION"] = "fp32"
    os.environ["FSDP_ALL_GATHER_PRECISION"] = "fp32"
    os.environ["FSDP_OPTIMIZER_STATE_PRECISION"] = "fp32"


def set_env(seed=0, latent_size=256):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    for _ in range(30):
        torch.randn(1, 4, latent_size, latent_size)


def get_dict_chunks(data, bs):
    keys = []
    for k in data:
        keys.append(k)
        if len(keys) == bs:
            yield keys
            keys = []
    if keys:
        yield keys


def create_tar(data_path):
    tar_path = f"{data_path}.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(data_path, arcname=os.path.basename(data_path))
    print(f"Created tar file: {tar_path}")
    return tar_path


def delete_directory(exp_name):
    if os.path.exists(exp_name):
        subprocess.run(["rm", "-r", exp_name], check=True)
        print(f"Deleted directory: {exp_name}")


@torch.inference_mode()
def visualize(config, args, model, items, bs, sample_steps, cfg_scale, pag_scale=1.0):
    if isinstance(items, dict):
        get_chunks = get_dict_chunks
    else:
        from diffusion.data.datasets.utils import get_chunks

    cur_seed = args.seed + int(rank)
    generator = torch.Generator(device=device).manual_seed(cur_seed)
    tqdm_desc = f"{save_root.split('/')[-1]}"
    # assert bs == 1  # current cfg not support multiple text during inference

    prompt_expander = None

    for prompt_batch in tqdm(prompts_dataloader, desc=tqdm_desc, unit="batch", leave=True):
        # data prepare
        prompts, hw, ar = (
            [],
            torch.tensor([[video_height, video_width]], dtype=torch.float, device=device).repeat(bs, 1),
            torch.tensor([[1.0]], device=device).repeat(bs, 1),
        )
        images = []
        if bs == 1:
            prompt = prompt_batch["prompt"][0]
            # prompt_clean, _, hw, ar, custom_hw = prepare_prompt_ar(prompt, base_ratios, device=device, show=False)
            # latent_size_h, latent_size_w = lat
            if config.task == "ti2v":
                prompt, image_path = prompt.split(image_split_token)
                images.append(image_path)
            prompts.append(prompt)

        else:
            for prompt in prompt_batch["prompt"]:
                if config.task == "ti2v":
                    prompt, image_path = prompt.split(image_split_token)
                    images.append(image_path)
                prompts.append(prompt)
            # latent_size_h, latent_size_w = latent_size, latent_size
        image_paths = images
        # check exists
        exist = False
        for i in range(bs):
            save_file_name = f"{prompt_batch['key'][i]}.mp4"
            save_path = os.path.join(save_root, save_file_name)
            exist = os.path.exists(save_path)
            if exist:
                break
        if exist:
            # make sure the noise is totally same
            torch.randn(
                bs,
                config.vae.vae_latent_dim,
                latent_temp,
                latent_height,
                latent_width,
                device=device,
                generator=generator,
            )
            continue

        if config.prompt_extend:
            with torch.no_grad():
                if config.task == "ti2v":
                    prompts = [
                        prompt_expander(prompt, tar_lang="en", image=image_paths[i], seed=cur_seed).prompt
                        for i, prompt in enumerate(prompts)
                    ]
                else:
                    prompts = [prompt_expander(prompt, tar_lang="en", seed=cur_seed).prompt for prompt in prompts]
            print(f"prompts: {prompts}")

        caption_embs = text_encoder(prompts)
        # text_encoder.to("cpu")
        torch.cuda.empty_cache()
        null_y = [null_caption_embs] * len(prompts)
        model_kwargs = dict(seq_len=video_pos_seq_len, block_mask=block_mask)
        # start sampling
        with torch.no_grad():

            n = len(prompts)
            z = torch.randn(
                n,
                config.vae.vae_latent_dim,
                latent_temp,
                latent_height,
                latent_width,
                device=device,
                generator=generator,
            ).to(weight_dtype)

            if config.task == "ti2v":
                images = [
                    read_image_from_path(
                        imgp,
                        (config.model.video_height, config.model.video_width),
                    )
                    for imgp in image_paths
                ]  # C,H,W
                image_embeds = encode_image(
                    config.image_encoder.image_encoder_type,
                    image_encoder,
                    torch.stack(images, dim=0),
                    device=device,
                )  # B,C,H,W
                if image_embeds is not None:
                    model_kwargs["clip_fea"] = image_embeds.repeat(2, 1, 1)
                # import ipdb; ipdb.set_trace()
                if config.model.image_latent_mode == "repeat":
                    image_context = vae_encode(
                        config.vae.vae_type, vae, torch.stack(images, dim=0)[:, :, None].to(vae_dtype), device=device
                    )  # 1,C,1,H,W
                    image_context = image_context.repeat(1, 1, latent_temp, 1, 1)  # B,C,F,H,W
                elif config.model.image_latent_mode == "zero":
                    image_context = vae_encode(
                        config.vae.vae_type, vae, torch.stack(images, dim=0)[:, :, None].to(vae_dtype), device=device
                    )  # 1,C,1,H,W
                    zero_context = torch.zeros_like(z)  # 1,C,F,H,W
                    length_image_context = image_context.shape[2]  # 1
                    zero_context[:, :, :length_image_context] = image_context
                    image_context = zero_context
                elif config.model.image_latent_mode == "video_zero":
                    batch_images = torch.stack(images, dim=0)[:, :, None]  # B,C,H,W -> B,C,1,H,W
                    dummy_vid = torch.zeros_like(batch_images).repeat(
                        1, 1, config.model.num_frames, 1, 1
                    )  # B,C,1,H,W -> B,C,F,H,W
                    dummy_vid[:, :, :1] = batch_images
                    # print(f"dummy_vid.shape: {dummy_vid.shape}")
                    with torch.no_grad():
                        image_context = vae_encode(
                            config.vae.vae_type, vae, dummy_vid.to(vae_dtype), device=accelerator.device
                        )  # 1,C,F,H,W
                # frame index mask for WanI2V 14B
                if config.model.mask == "first":
                    msk = torch.ones(1, config.model.num_frames, latent_height, latent_width, device=device)
                    msk[:, 1:] = 0
                    if config.vae.vae_type == "WanVAE":
                        msk = torch.concat(
                            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
                        )  # 1,84,h,w
                        # print(f"msk.shape: {msk.shape}")

                    msk = msk.view(1, msk.shape[1] // 4, 4, latent_height, latent_width)  # 1,21,4,h,w
                    msk = msk.transpose(1, 2)  # 1,4,21,h,w
                    image_context = torch.cat([msk, image_context], dim=1)  # 1,C+4,f,h,w
                    # print(f"image_context.shape: {image_context.shape}")
                # import ipdb; ipdb.set_trace()
                model_kwargs["y"] = torch.cat([image_context, image_context], dim=0)  # B,C,F,H,W

            # model_kwargs = dict(data_info={"img_hw": hw, "aspect_ratio": ar}, mask=emb_masks)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                if args.sampling_algo == "dpm-solver":
                    dpm_solver = DPMS(
                        model,
                        condition=caption_embs,
                        uncondition=null_y,
                        cfg_scale=cfg_scale,
                        model_kwargs=model_kwargs,
                    )
                    samples = dpm_solver.sample(
                        z,
                        steps=sample_steps,
                        order=2,
                        skip_type="time_uniform",
                        method="multistep",
                    )
                elif args.sampling_algo == "sa-solver":
                    sa_solver = SASolverSampler(model.forward_with_dpmsolver, device=device)
                    samples = sa_solver.sample(
                        S=25,
                        batch_size=n,
                        shape=(config.vae.vae_latent_dim, latent_temp, latent_height, latent_width),
                        eta=1,
                        conditioning=caption_embs,
                        unconditional_conditioning=null_y,
                        unconditional_guidance_scale=cfg_scale,
                        model_kwargs=model_kwargs,
                    )[0]
                elif args.sampling_algo == "flow_euler":
                    flow_solver = FlowEuler(
                        model,
                        condition=caption_embs,
                        uncondition=null_y,
                        cfg_scale=cfg_scale,
                        flow_shift=flow_shift,
                        model_kwargs=model_kwargs,
                    )
                    samples = flow_solver.sample(
                        z,
                        steps=sample_steps,
                    )
                elif args.sampling_algo == "flow_dpm-solver":
                    dpm_solver = DPMS(
                        model,
                        condition=caption_embs,
                        uncondition=null_y,
                        guidance_type=guidance_type,
                        cfg_scale=cfg_scale,
                        pag_scale=pag_scale,
                        pag_applied_layers=pag_applied_layers,
                        model_type="flow",
                        model_kwargs=model_kwargs,
                        schedule="FLOW",
                        interval_guidance=args.interval_guidance,
                        condition_as_list=True,
                    )
                    samples = dpm_solver.sample(
                        z,
                        steps=sample_steps,
                        order=2,
                        skip_type="time_uniform_flow",
                        method="multistep",
                        flow_shift=flow_shift,
                    )
                elif args.sampling_algo == "unipc":
                    flow_solver = UniPC(
                        model,
                        condition=caption_embs,
                        uncondition=null_y,
                        cfg_scale=cfg_scale,
                        flow_shift=flow_shift,
                        model_kwargs=model_kwargs,
                        condition_as_list=True,
                    )
                    samples = flow_solver.sample(z, steps=sample_steps)
                else:
                    raise ValueError(f"{args.sampling_algo} is not defined")

        torch.cuda.synchronize()
        accelerator.wait_for_everyone()

        samples = samples.to(vae_dtype)
        samples = vae_decode(config.vae.vae_type, vae, samples)
        if isinstance(samples, list):
            samples = torch.stack(samples, dim=0)  # [3,F,H,W]*B -> [B,3,F,H,W]
        videos = (
            torch.clamp(127.5 * samples + 127.5, 0, 255).permute(0, 2, 3, 4, 1).to("cpu", dtype=torch.uint8)
        )  # B,C,T,H,W -> B,T,H,W,C

        torch.cuda.empty_cache()

        os.umask(0o000)
        for i, video in enumerate(videos):
            # skip pad samples
            if prompt_batch["key"][i].startswith("<pad>"):
                continue
            if video.size(0) == 1:
                # save image
                save_file_name = f"{prompt_batch['key'][i]}.png"
                save_path = os.path.join(save_root, save_file_name)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                imageio.imwrite(save_path, video[0].numpy())
            else:
                # save video
                save_file_name = f"{prompt_batch['key'][i]}.mp4"
                save_path = os.path.join(save_root, save_file_name)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # save_image(sample, save_path, nrow=1, normalize=True, value_range=(-1, 1))
                writer = imageio.get_writer(save_path, fps=16, codec="libx264", quality=8)
                for frame in video.numpy():
                    writer.append_data(frame)
                writer.close()

            if args.save_qkv:
                prompt_key_name = prompt_batch["key"][i].replace(" ", "_")
                qkv_save_path = os.path.join(save_root, f"qkv_{prompt_key_name}")
                os.makedirs(qkv_save_path, exist_ok=True)
                for tt, t_buffer in model.qkv_store_buffer.items():
                    for block_idx, block_buffer in t_buffer.items():
                        torch.save(block_buffer, os.path.join(qkv_save_path, f"t{tt}_{block_idx}.pth"))
            if args.save_block_output:
                prompt_key_name = prompt_batch["key"][i].replace(" ", "_")
                block_output_save_path = os.path.join(save_root, f"block_output_{prompt_key_name}")
                os.makedirs(block_output_save_path, exist_ok=True)
                # Access the underlying model
                underlying_model = model
                if hasattr(model, "module"):
                    underlying_model = model.module
                elif hasattr(model, "_fsdp_wrapped_module"):
                    underlying_model = model._fsdp_wrapped_module

                # Save outputs with rank information
                if hasattr(underlying_model, "block_output_buffer"):
                    for tt, t_buffer in underlying_model.block_output_buffer.items():
                        for block_idx, block_buffer in t_buffer.items():
                            # print(f"block_buffer: {block_buffer}")
                            torch.save(block_buffer, os.path.join(block_output_save_path, f"t{tt}_{block_idx}.pth"))

                # Wait for all processes to finish saving
                accelerator.wait_for_everyone()

        # import ipdb; ipdb.set_trace()

    flush()
    torch.cuda.synchronize()
    accelerator.wait_for_everyone()

    # Clean up distributed processes
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    return


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config")
    return parser.parse_known_args()[0]


@dataclass
class WanInference(WanConfig):
    config: Optional[str] = "mhla_videogen/configs/Wan_1300M_came8bit_fsdp_mhla.yaml"  # config
    model_path: Optional[str] = None
    work_dir: Optional[str] = None
    version: str = "wanx2.1"
    txt_file: str = None
    json_file: Optional[str] = None
    sample_nums: int = 100_000
    bs: int = 1
    cfg_scale: float = 3.0
    pag_scale: float = 1.0
    sampling_algo: str = "flow_dpm-solver"
    seed: int = 0
    dataset: str = "custom"
    step: int = 50
    add_label: str = ""
    tar_and_del: bool = False
    exist_time_prefix: str = ""
    gpu_id: int = 0
    custom_video_size: Optional[int] = None
    start_index: int = 0
    end_index: int = 30_000
    interval_guidance: List[float] = field(default_factory=lambda: [0, 1])
    ablation_selections: Optional[List[float]] = None
    ablation_key: Optional[str] = None
    debug: bool = False
    if_save_dirname: bool = False
    vbench_prompt_path: Optional[str] = None  # can be path to vbench_prompts or vbench_prompts_extended
    image_split_token: str = "<image>"
    prompt_extend: bool = False
    prompt_extend_model: Optional[str] = None
    mixed_precision: Optional[str] = None
    save_qkv: bool = False
    save_block_output: bool = False
    t5_fsdp: bool = False
    flow_shift: Optional[float] = None
    negative_prompt: str = ""


def load_vbench_prompts(path: str) -> dict:
    """Load vbench prompts from a directory containing txt files.

    Args:
        path (str): Path to the directory containing txt files, where each file name represents a dimension

    Returns:
        dict: Dictionary with keys in format "{dimension}/{prompt}-{index}" and values as prompt
    """
    # vbench origin prompts
    vbench_origin_prompts_path = os.path.join(os.path.dirname(path), "vbench_prompts")

    prompts_dict = {}

    # Get all txt files in the directory
    txt_files = sorted([f for f in os.listdir(path) if f.endswith(".txt")])

    for txt_file in txt_files:
        # Get dimension name from filename (remove .txt extension)
        dimension = os.path.splitext(txt_file)[0]
        file_path = os.path.join(path, txt_file)

        # Read prompts from txt file
        with open(file_path) as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        vbench_original_file_path = os.path.join(vbench_origin_prompts_path, txt_file)
        with open(vbench_original_file_path) as f:
            vbench_original_lines = [line.strip() for line in f.readlines() if line.strip()]

        assert len(lines) == len(
            vbench_original_lines
        ), f"Number of prompts in {txt_file} and {vbench_original_file_path} are different"

        # Create 5 entries for each prompt with indices 0-4
        for i, line in enumerate(lines):
            prompt = line  # this is the caption used for generation
            for idx in range(5):
                key = f"{dimension}/{vbench_original_lines[i]}-{idx}"  # this is the key used for saving
                prompts_dict[key] = prompt

    return prompts_dict


class DistributePromptsDataset(torch.utils.data.Dataset):
    """Dataset for vbench inference.

    Args:
        prompts: Dictionary with keys and (prompt, image_path) tuples as values
    """

    def __init__(self, prompts):
        if isinstance(prompts, dict):
            self.prompts = prompts
        else:
            # Convert list to dict where key and value are the same
            self.prompts = {prompt: prompt for prompt in prompts}
        self.keys_list = list(self.prompts.keys())

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        key = self.keys_list[idx]
        prompt = self.prompts[key]
        return {
            "key": key,
            "prompt": prompt,
        }


if __name__ == "__main__":

    args = get_args()
    config = args = pyrallis.parse(config_class=WanInference, config_path=args.config)
    image_split_token = config.image_split_token
    args.video_size = (config.model.video_height, config.model.video_width)
    if args.custom_video_size:
        args.video_size = args.custom_video_size
        print(f"custom_video_size: {args.custom_video_size}")

    video_width = config.model.video_width
    video_height = config.model.video_height
    num_frames = config.model.num_frames

    latent_width = int(video_width) // config.vae.vae_stride[2]
    latent_height = int(video_height) // config.vae.vae_stride[1]
    latent_temp = int(config.model.num_frames - 1) // config.vae.vae_stride[0] + 1

    video_pos_seq_len = math.ceil(
        (latent_height * latent_width) / (config.model.patch_size[1] * config.model.patch_size[2]) * latent_temp
    )
    set_env(args.seed, latent_height)
    if config.train.use_fsdp:
        set_fsdp_env()

    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=300)  # change timeout to avoid a strange NCCL bug

    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision if config.mixed_precision else "no",
        kwargs_handlers=[init_handler],
    )

    device = accelerator.device
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    logger = get_root_logger()

    # if config.train.use_fsdp and config.prompt_extend:
    #     logger.warning("FSDP is not supported for prompt extend, using single GPU or DDP")
    #     config.prompt_extend = False

    # only support fixed latent size currently
    # latent_size = args.image_size // config.vae.vae_downsample_rate
    max_sequence_length = config.text_encoder.text_len
    flow_shift = config.scheduler.flow_shift if args.flow_shift is None else args.flow_shift
    pag_applied_layers = None
    guidance_type = "classifier-free_PAG"
    assert (
        isinstance(args.interval_guidance, list)
        and len(args.interval_guidance) == 2
        and args.interval_guidance[0] <= args.interval_guidance[1]
    )
    args.interval_guidance = [max(0, args.interval_guidance[0]), min(1, args.interval_guidance[1])]
    sample_steps_dict = {"dpm-solver": 20, "sa-solver": 25, "flow_dpm-solver": 20, "flow_euler": 28}
    sample_steps = args.step if args.step != -1 else sample_steps_dict[args.sampling_algo]

    # weight_dtype = get_weight_dtype(config.model.mixed_precision)
    weight_dtype = get_weight_dtype(config.mixed_precision) if config.mixed_precision else torch.float32

    vae_dtype = get_weight_dtype(config.vae.weight_dtype)
    vae = get_vae(config.vae.vae_type, config.vae.vae_pretrained, device, dtype=vae_dtype, config=config.vae)

    text_encoder = T5EncoderModel(
        text_len=config.text_encoder.text_len,
        dtype=get_weight_dtype(config.text_encoder.t5_dtype),
        device=device,
        checkpoint_path=config.text_encoder.t5_checkpoint,
        tokenizer_path=config.text_encoder.t5_tokenizer,
        shard_fn=partial(shard_model, device_id=local_rank)
        if config.train.use_fsdp or config.t5_fsdp
        else None,  # partial(shard_model, device_id=local_rank)
    )

    null_caption_embs = text_encoder(config.negative_prompt)[0]

    image_encoder = None
    if config.image_encoder is not None and config.image_encoder.image_encoder_type is not None:
        image_encoder = get_image_encoder(
            config.image_encoder.image_encoder_type,
            config.image_encoder.image_encoder_pretrained,
            config.image_encoder.image_encoder_tokenizer,
            accelerator.device,
            dtype=get_weight_dtype(config.image_encoder.weight_dtype),
            config=config.image_encoder,
        )
    # import ipdb; ipdb.set_trace()
    if config.model.from_pretrained is None:
        model_configs = init_model_configs(config.model, config.vae)
        model = WanLinearAttentionModel(**model_configs).eval()

    elif config.model.from_pretrained.endswith(".json"):
        model = WanModel.from_config(
            config.model.from_pretrained,
        ).eval()
    elif os.path.isdir(config.model.from_pretrained) and os.path.exists(
        os.path.join(config.model.from_pretrained, "config.json")
    ):
        model = WanModel.from_pretrained(
            config.model.from_pretrained,
        ).eval()

    # add lora
    use_lora = hasattr(config, "lora") and config.lora is not None and config.lora.use_lora
    if use_lora:
        logger.info("Applying LoRA to model")
        # Step 2: Apply LoRA configuration
        peft_lora_config = LoraConfig(
            r=config.lora.rank,
            lora_alpha=config.lora.alpha,
            target_modules=config.lora.target_modules,
            lora_dropout=config.lora.dropout,
            bias=getattr(config.lora, "bias", "none"),
            init_lora_weights=getattr(config.lora, "init_lora_weights", True),
            fan_in_fan_out=getattr(config.lora, "fan_in_fan_out", False),
        )
        # import ipdb; ipdb.set_trace()
        # Apply LoRA to the model
        model = get_peft_model(model, peft_lora_config)

    if args.model_path:
        logger.info("Generating sample from ckpt: %s" % args.model_path)
        state_dict = find_model(args.model_path)
    else:
        state_dict = {
            "state_dict": model.state_dict(),
        }
        args.model_path = ""

    if args.model_path and args.model_path.endswith(".bin"):
        logger.info("Loading fsdp bin checkpoint....")
        old_state_dict = state_dict
        state_dict = dict()
        state_dict["state_dict"] = old_state_dict

    missing, unexpected = model.load_state_dict(state_dict["state_dict"], strict=True)
    logger.warning(f"Missing keys: {missing}")
    logger.warning(f"Unexpected keys: {unexpected}")

    if args.save_qkv:
        # assert config.train.use_fsdp == False, "FSDP is not supported for saving qkv"
        model.save_qkv = True
        model.qkv_store_buffer = {}
    if args.save_block_output:
        # assert config.train.use_fsdp == False, "FSDP is not supported for saving block output"
        model.save_block_output = True
        model.block_output_buffer = {}
        model.register_block_hook(device="cpu", score_only=True)

    block_mask = None
    # Aura Attention Design # TODO: add a reference here
    if config.model.attn_mask == "diagonal":
        # import ipdb; ipdb.set_trace()
        from matplotlib import pyplot as plt

        from tools.attn_mask.gen_nlogn_mask import gen_log_mask_shrinked

        flex_attn_padding = 128
        os.makedirs(config.model.diagonal_mask_root, exist_ok=True)
        diagonal_mask_path = f"{config.model.diagonal_mask_root}/nlogn_mask_h{int(latent_height/model.patch_size[-2])}_w{int(latent_width/model.patch_size[-1])}_f{int(latent_temp/model.patch_size[0])}_p{flex_attn_padding}_b{config.model.diagonal_block_size}.pt"
        if not os.path.exists(diagonal_mask_path):
            logger.info(f"Creating diagonal mask for {diagonal_mask_path}")
            num_token = (
                int(latent_width / model.patch_size[-1])
                * int(latent_height / model.patch_size[-2])
                * int(latent_temp / model.patch_size[0])
            )
            num_token_padded = ((num_token + flex_attn_padding - 1) // flex_attn_padding) * flex_attn_padding
            num_frames = int(latent_temp / model.patch_size[0])
            block_size = config.model.diagonal_block_size
            diagonal_mask = model.create_diagonal_mask(num_token_padded, num_token, num_frames, block_size)
            if accelerator.is_main_process:
                plt.imshow(diagonal_mask.cpu().numpy()[:, :], cmap="hot")
                plt.colorbar()
                plt.title("diagonal mask")
                plt.savefig(diagonal_mask_path.replace(".pt", ".png"))
                plt.close()
                logger.info(
                    f"Saving diagonal mask image to {config.work_dir}/nlogn_mask_h{int(latent_height/model.patch_size[-2])}_w{int(latent_width/model.patch_size[-1])}_f{int(latent_temp/model.patch_size[0])}_p{flex_attn_padding}_b{config.model.diagonal_block_size}.png"
                )
                logger.info(f"Saving diagonal mask to {diagonal_mask_path}")
                torch.save(diagonal_mask, diagonal_mask_path)
    elif config.model.attn_mask == "radial_sparse":
        # import ipdb; ipdb.set_trace()
        # radial sparse attn, requires the number of tokens to be a multiple of 128
        from matplotlib import pyplot as plt

        from tools.attn_mask.gen_nlogn_mask import gen_log_mask_shrinked

        block_size = 128
        os.makedirs(config.model.diagonal_mask_root, exist_ok=True)
        block_mask = {}
        lw, lh, lt = latent_width, latent_height, latent_temp
        diagonal_mask_path = f"{config.model.diagonal_mask_root}/nlogn_radial_mask_h{int(lh/model.patch_size[-2])}_w{int(lw/model.patch_size[-1])}_f{int(lt/model.patch_size[0])}.pt"
        if os.path.exists(diagonal_mask_path):
            logger.info(f"Loading radial sparse mask from {diagonal_mask_path}")
            cur_mask = torch.load(diagonal_mask_path, map_location="cpu")
            block_mask = cur_mask
        else:
            # create new mask
            logger.info(f"Creating radial sparse mask for {diagonal_mask_path}")
            num_token = int(lw / model.patch_size[-1]) * int(lh / model.patch_size[-2]) * int(lt / model.patch_size[0])
            assert num_token % block_size == 0, f"Number of tokens must be a multiple of {block_size}"
            num_frames = int(lt / model.patch_size[0])
            diagonal_mask = gen_log_mask_shrinked(num_token, num_token, num_frames, block_size).bool()
            block_mask = diagonal_mask
            if accelerator.is_main_process:
                plt.imshow(diagonal_mask.cpu().numpy()[:, :], cmap="hot")
                plt.colorbar()
                plt.title("radial sparse mask")
                plt.savefig(diagonal_mask_path.replace(".pt", ".png"))
                plt.close()
                logger.info(f"Saving radial sparse mask image to {diagonal_mask_path.replace('.pt', '.png')}")
                logger.info(f"Saving radial sparse mask to {diagonal_mask_path}")
                torch.save(diagonal_mask, diagonal_mask_path)

        # block_mask = diagonal_mask
        model.diagonal_mask = None

    # model.eval().to(device, weight_dtype)
    # base_ratios = eval(f"ASPECT_RATIO_{args.image_size}_TEST")
    args.sampling_algo = (
        args.sampling_algo
        if ("flow" not in args.model_path or args.sampling_algo == "flow_dpm-solver")
        else "flow_euler"
    )

    if args.work_dir is None:
        work_dir = (
            f"/{os.path.join(*args.model_path.split('/')[:-2])}"
            if args.model_path.startswith("/")
            else os.path.join(*args.model_path.split("/")[:-2])
        )
    else:
        work_dir = args.work_dir
    config.work_dir = work_dir
    img_save_dir = os.path.join(str(work_dir), "vis")

    logger.info(colored(f"Saving images at {img_save_dir}", "green"))

    if args.vbench_prompt_path is not None and args.dataset == "vbench":
        prompts_dict = load_vbench_prompts(args.vbench_prompt_path)

    elif args.json_file is not None:
        prompts_dict = json.load(open(args.json_file))
    else:
        with open(args.txt_file) as f:
            prompts = [item.strip() for item in f.readlines()]
        prompts_dict = {}
        for prompt in prompts:
            # split prompt and image path
            if image_split_token in prompt:
                prompt_text, image_path = prompt.split(image_split_token)
                prompts_dict[prompt_text[:100]] = prompt
            else:
                prompts_dict[prompt[:100]] = prompt

    if args.debug:
        prompts = [
            'a blackboard wrote text "Hello World"'
            'Text" Super Dad Mode ON", t shirt design, This is a graffiti-style image.The letters are surrounded by a playful, abstract design of paw prints and pet-related shapes, such as a heart-shaped bone and a cat-whisker-shaped element.',
            '"NR Beauty Hair" logo para peluqueria, product, typography, fashion, painting',
            'Text"Goblins gone wild.", The text is written in an elegant, vintage-inspired font and each letter in the text showed in different colors.',
            "An awe-inspiring 3D render of the mahir Olympics logo, set against the backdrop of a fiery, burning Olympic flame. The flames dance and intertwine to form the iconic Olympic rings and typography, while the Eiffel Tower stands tall in the distance. The cinematic-style poster is rich in color and detail, evoking a sense of excitement and anticipation for the upcoming games., ukiyo-e, vibrant, cinematic, 3d render, typography, poster",
            'Cute cartoon back style of a couple, wearing a black t shirts , she have long hair with the name "C". He have staright hair and light beard with the name "J"white color,heart snowy atmosphere, typography, 3d render, portrait photography, fashion',
            'A captivating 3D render of a whimsical, colorful scene, featuring the word "Muhhh" spelled out in vibrant, floating balloons. The wordmark hovers above a lush, emerald green field. A charming, anthropomorphic rabbit with a wide smile and twinkling eyes hops alongside the balloon letters. The background showcases a serene, dreamy sky with soft pastel hues, creating an overall atmosphere of joy, enchantment, and surrealism. The 3D render is a stunning illustration that blends fantasy and realism effortlessly., illustration, 3d render',
            'create a logo for a company named "FUN"',
            "A stunningly realistic image of an Asian woman sitting on a plush sofa, completely engrossed in a book. She is wearing cozy loungewear and has headphones on, indicating her desire for a serene and quiet environment. In one hand, she holds a can of water, providing a refreshing sensation. The adjacent table features an array of snacks and books, adding to the cozy ambiance of the scene. The room is filled with natural light streaming through vibrantly decorated windows, and tasteful decorations contribute to the overall relaxing and soothing atmosphere.",
            'A captivating 3D logo illustration of the name "ANGEL" in a romantic and enchanting Follow my Page poster design. The lettering is adorned with a majestic, shimmering crown encrusted with intricate gemstones. Swirling pink and purple patterns, reminiscent of liquid or air, surround the crown, with beautiful pink flowers in full bloom and bud adorning the design. Heart-shaped decorations enhance the romantic ambiance, and a large, iridescent butterfly with intricate wings graces the right side of the crown. The muted purple background contrasts with the bright and lively elements within the composition, creating a striking visual effect. The 3D rendering showcases the intricate details and depth of the design, making it a truly mesmerizing piece of typography, 3D render, and illustration art., illustration, typography, poster, 3d render',
            'A human wearing a T-shirt with Text "NVIDIA" and logo',
            'Logo with text "Hi"',
        ]
        prompts_dict = {v[:100]: v for v in prompts}

    items = list(prompts_dict.keys())

    items = items[: max(0, args.sample_nums)]
    items = items[max(0, args.start_index) : min(len(items), args.end_index)]
    new_prompts_dict = {}
    for item in items:
        new_prompts_dict[item] = prompts_dict[item]

    # pad the new prompts dict to be divided by world size
    pad_samples = 0
    total_samples = len(new_prompts_dict)
    world_size = get_world_size()
    if total_samples % world_size != 0:
        pad_samples = world_size - total_samples % world_size
        for i in range(pad_samples):
            new_prompts_dict[f"<pad>_{i}"] = new_prompts_dict[items[0]]

    logger.info(
        f"Eval {min(args.sample_nums, len(new_prompts_dict))}/{len(new_prompts_dict)} samples, with {pad_samples} pad samples"
    )

    prompts_dataset = DistributePromptsDataset(new_prompts_dict)
    prompts_dataloader = torch.utils.data.DataLoader(prompts_dataset, batch_size=args.bs, shuffle=False)
    # prepare dataloader, model, text encoder
    prompts_dataloader, model = accelerator.prepare(prompts_dataloader, model)

    match = re.search(r".*epoch_(\d+).*step_(\d+).*", args.model_path)
    epoch_name, step_name = match.groups() if match else ("unknown", "unknown")

    os.umask(0o000)
    os.makedirs(img_save_dir, exist_ok=True)
    logger.info(f"Sampler {args.sampling_algo}")

    def create_save_root(args, dataset, epoch_name, step_name, sample_steps, guidance_type):
        save_root = os.path.join(
            img_save_dir,
            f"{dataset}_epoch{epoch_name}_step{step_name}_scale{args.cfg_scale}"
            f"_step{sample_steps}_{num_frames}x{video_width}x{video_height}_bs{args.bs}_samp{args.sampling_algo}"
            f"_seed{args.seed}_{str(weight_dtype).split('.')[-1]}",
        )

        if args.pag_scale != 1.0:
            save_root = save_root.replace(f"scale{args.cfg_scale}", f"scale{args.cfg_scale}_pagscale{args.pag_scale}")
        if flow_shift != 1.0:
            save_root += f"_flowshift{flow_shift}"
        if guidance_type != "classifier-free":
            save_root += f"_{guidance_type}"
        if args.interval_guidance[0] != 0 and args.interval_guidance[1] != 1:
            save_root += f"_intervalguidance{args.interval_guidance[0]}{args.interval_guidance[1]}"

        save_root += f"_imgnums{args.sample_nums}" + args.add_label
        return save_root

    def guidance_type_select(default_guidance_type, pag_scale, attn_type):
        guidance_type = default_guidance_type
        if not (pag_scale > 1.0 and attn_type == "linear"):
            logger.info("Setting back to classifier-free")
            guidance_type = "classifier-free"
        return guidance_type

    dataset = args.dataset

    # guidance_type = guidance_type_select(guidance_type, args.pag_scale, "")
    guidance_type = "classifier-free"
    logger.info(f"Inference with {weight_dtype}, guidance_type: {guidance_type}, flow_shift: {flow_shift}")

    save_root = create_save_root(args, dataset, epoch_name, step_name, sample_steps, guidance_type)
    os.makedirs(save_root, exist_ok=True)
    if args.if_save_dirname and args.gpu_id == 0:
        os.makedirs(f"{work_dir}/metrics", exist_ok=True)
        # save at work_dir/metrics/tmp_xxx.txt for metrics testing
        with open(f"{work_dir}/metrics/tmp_{dataset}_{time.time()}.txt", "w") as f:
            print(f"save tmp file at {work_dir}/metrics/tmp_{dataset}_{time.time()}.txt")
            f.write(os.path.basename(save_root))

    visualize(
        config=config,
        args=args,
        model=model,
        items=items,
        bs=args.bs,
        sample_steps=sample_steps,
        cfg_scale=args.cfg_scale,
        pag_scale=args.pag_scale,
    )

    if args.tar_and_del:
        create_tar(save_root)
        delete_directory(save_root)

    print(
        colored(f"Sana inference has finished. Results stored at ", "green"),
        colored(f"{img_save_dir}", attrs=["bold"]),
        ".",
    )
