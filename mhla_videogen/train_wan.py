# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import datetime
import gc
import hashlib
import json
import os
import os.path as osp
import time
import warnings
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from dotenv import load_dotenv

warnings.filterwarnings("ignore")  # ignore warning

import math

import imageio
import pyrallis
import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs, skip_first_batches

# LoRA imports
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from termcolor import colored

from diffusion import DPMS, FlowEuler, Scheduler
from diffusion.data.builder import build_dataloader, build_dataset
from diffusion.data.transforms import read_image_from_path
from diffusion.data.wids import DistributedRangedSampler
from diffusion.model.builder import (
    encode_image,
    get_image_encoder,
    get_vae,
    vae_decode,
    vae_encode,
)
from diffusion.model.respace import compute_density_for_timestep_sampling
from diffusion.model.utils import get_weight_dtype
from diffusion.model.wan import T5EncoderModel, WanLinearAttentionModel, WanModel, init_model_configs
from diffusion.model.wan.model import WanLinearAttention
from diffusion.utils.checkpoint import load_checkpoint, save_checkpoint
from diffusion.utils.config import one_logger_callback_config
from diffusion.utils.config_wan import WanConfig
from diffusion.utils.data_sampler import AspectRatioBatchSamplerVideo
from diffusion.utils.dist_utils import flush, get_world_size
from diffusion.utils.logger import LogBuffer, get_root_logger
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import DebugUnderflowOverflow, init_random_seed, set_random_seed
from diffusion.utils.optimizer import auto_scale_lr, build_optimizer
# import ipdb

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import inspect

print(inspect.getfile(WanLinearAttentionModel))

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

    # NOTE: if set to False for PEFT models, it may save memory, but requires to modify the auto_wrap_policy:
    # https://huggingface.co/docs/peft/en/accelerate/fsdp
    os.environ["FSDP_USE_ORIG_PARAMS"] = "true"

    # Sharding strategy
    # [1] FULL_SHARD (shards optimizer states, gradients and parameters), [2] SHARD_GRAD_OP (shards optimizer states and gradients), [3] NO_SHARD (DDP), [4] HYBRID_SHARD (shards optimizer states, gradients and parameters within each node while each node has full copy), [5] HYBRID_SHARD_ZERO2 (shards optimizer states and gradients within each node while each node has full copy).
    os.environ["FSDP_SHARDING_STRATEGY"] = "HYBRID_SHARD"

    # Memory optimization settings (optional)
    os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "false"  # "false"
    os.environ["FSDP_OFFLOAD_PARAMS"] = "false"  # "false"

    # Precision settings
    os.environ["FSDP_REDUCE_SCATTER_PRECISION"] = "fp32"
    os.environ["FSDP_ALL_GATHER_PRECISION"] = "fp32"
    os.environ["FSDP_OPTIMIZER_STATE_PRECISION"] = "fp32"


def ema_update(model_dest, model_src, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


@torch.inference_mode()
def log_validation(accelerator, config, model, logger, step, device, vae=None, init_noise=None):
    torch.cuda.empty_cache()
    vis_sampler = config.scheduler.vis_sampler
    # model = accelerator.unwrap_model(model).eval()
    hw = torch.tensor([[video_height, video_width]], dtype=torch.float, device=device).repeat(1, 1)
    ar = torch.tensor([[1.0]], device=device).repeat(1, 1)
    null_y = torch.load(null_embed_path, map_location="cpu")
    null_y = null_y["uncond_prompt_embeds"].to(device)

    # Create sampling noise:
    logger.info("Running validation... ")
    video_logs = []

    if config.train.offload_vae:
        vae.to(device)

    def run_sampling(init_z=None, label_suffix="", vae=None, sampler="dpm-solver"):
        latents = []
        current_video_logs = []
        for prompt_idx, prompt in enumerate(validation_prompts):
            # one_logger_callbacks.on_validation_batch_start()
            z = (
                torch.randn(1, config.vae.vae_latent_dim, latent_temp, latent_height, latent_width, device=device)
                if init_z is None
                else init_z
            )
            embed = torch.load(
                osp.join(config.train.valid_prompt_embed_root, f"{prompt[:50]}_{valid_prompt_embed_suffix}"),
                map_location="cpu",
            )
            # caption_embs, emb_masks = embed["caption_embeds"].to(device), embed["emb_mask"].to(device)
            # model_kwargs = dict(data_info={"img_hw": hw, "aspect_ratio": ar}, mask=emb_masks)
            caption_embs = embed["caption_embeds"].to(device)
            # for multi-scale training, the seq length is different for each batch
            video_pos_seq_len = math.ceil(
                (z.size(3) * z.size(4))
                / (config.model.patch_size[1] * config.model.patch_size[2])
                * z.size(2)
                / config.model.patch_size[0]
            )
            if isinstance(block_mask, dict):
                cur_block_mask = block_mask[f"{z.size(2)}x{z.size(3)}x{z.size(4)}"]
            else:
                cur_block_mask = block_mask
            if cur_block_mask is not None:
                cur_block_mask = cur_block_mask.to(device)
            model_kwargs = dict(
                data_info={"img_hw": hw, "aspect_ratio": ar}, seq_len=video_pos_seq_len, block_mask=cur_block_mask
            )
            if config.task == "ti2v":

                image_context = embed["image_context"].to(device)  # 1,C,1,H,W
                if config.model.image_latent_mode == "repeat":
                    image_context = image_context.repeat(1, 1, latent_temp, 1, 1)  # B,C,F,H,W
                elif config.model.image_latent_mode == "zero":
                    zero_context = torch.zeros_like(z)  # 1,C,F,H,W
                    length_image_context = image_context.shape[2]  # 1
                    zero_context[:, :, :length_image_context] = image_context
                    image_context = zero_context
                elif config.model.image_latent_mode == "video_zero":
                    pass  # already same shape with latent

                # frame index mask for WanI2V 14B
                if config.model.mask == "first":
                    bs = image_context.size(0)
                    msk = torch.ones(bs, config.model.num_frames, latent_height, latent_width, device=device)
                    msk[:, 1:] = 0
                    if config.vae.vae_type == "WanVAE":
                        msk = torch.concat(
                            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
                        )  # 1,84,h,w
                        # print(f"msk.shape: {msk.shape}")

                    msk = msk.view(bs, msk.shape[1] // 4, 4, latent_height, latent_width)  # 1,21,4,h,w
                    msk = msk.transpose(1, 2)  # 1,4,21,h,w
                    image_context = torch.cat([msk, image_context], dim=1)  # 1,C+4,f,h,w
                    # image_context = image_context.repeat(1, 1, latent_temp, 1, 1)  # B,C,F,H,W

                # NOTE hard code to support ti2v
                image_context = torch.cat([image_context, image_context], dim=0)  # 2,C,1,H,W
                model_kwargs["y"] = image_context  # B,C,F,H,W

                image_embeds = embed.get("image_embeds", None)
                if image_embeds is not None:
                    image_embeds = image_embeds.to(device)
                    model_kwargs["clip_fea"] = torch.cat([image_embeds, image_embeds], dim=0)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                # ipdb.set_trace()
                if sampler == "dpm-solver":
                    dpm_solver = DPMS(
                        model,  # model.forward_with_dpmsolver, it is the same as model forward if not pred_sigma
                        condition=caption_embs,
                        uncondition=null_y,
                        cfg_scale=config.cfg_scale,
                        model_kwargs=model_kwargs,
                        condition_as_list=True,
                    )
                    denoised = dpm_solver.sample(
                        z,
                        steps=14,
                        order=2,
                        skip_type="time_uniform",
                        method="multistep",
                    )
                elif sampler == "flow_euler":
                    flow_solver = FlowEuler(
                        model,
                        condition=caption_embs,
                        uncondition=null_y,
                        cfg_scale=config.cfg_scale,
                        model_kwargs=model_kwargs,
                    )
                    denoised = flow_solver.sample(z, steps=28)
                elif sampler == "flow_dpm-solver":
                    # ipdb.set_trace()
                    dpm_solver = DPMS(
                        model,  # model.forward_with_dpmsolver, it is the same as model forward if not pred_sigma
                        condition=caption_embs,
                        uncondition=null_y,
                        cfg_scale=config.cfg_scale,
                        model_type="flow",
                        model_kwargs=model_kwargs,
                        schedule="FLOW",
                        condition_as_list=True,
                    )
                    denoised = dpm_solver.sample(
                        z,
                        steps=20,
                        order=2,
                        skip_type="time_uniform_flow",
                        method="multistep",
                        flow_shift=config.scheduler.flow_shift,
                    )
                else:
                    raise ValueError(f"{sampler} not implemented")

            latents.append(denoised)
            # one_logger_callbacks.on_validation_batch_end()

        torch.cuda.empty_cache()

        if vae is None:
            vae = get_vae(
                config.vae.vae_type, config.vae.vae_pretrained, accelerator.device, dtype=vae_dtype, config=config.vae
            ).to(vae_dtype)

        for prompt, latent in zip(validation_prompts, latents):
            latent = latent.to(vae_dtype)
            # samples = vae.decode(latent)  # List[3,F,H,W]*1
            samples = vae_decode(config.vae.vae_type, vae, latent)
            video = (
                torch.clamp(127.5 * samples[0] + 127.5, 0, 255).permute(1, 0, 2, 3).to("cpu", dtype=torch.uint8).numpy()
            )  # C,T,H,W -> T,C,H,W
            current_video_logs.append({"validation_prompt": prompt + label_suffix, "videos": video})

        return current_video_logs

    # First run with original noise
    video_logs += run_sampling(init_z=None, label_suffix="", vae=vae, sampler=vis_sampler)

    # Second run with init_noise if provided
    if init_noise is not None:
        torch.cuda.empty_cache()
        gc.collect()
        init_noise = torch.clone(init_noise).to(device)
        video_logs += run_sampling(init_z=init_noise, label_suffix=" w/ init noise", vae=vae, sampler=vis_sampler)

    if accelerator.is_main_process:
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                pass
                # NOTE tensorboard does not support video logging
                # for validation_prompt, image in formatted_images:
                # tracker.writer.add_images(validation_prompt, image[None, ...], step, dataformats="NHWC")
            elif tracker.name == "wandb":
                # NOTE wandb shows media is not installed
                import wandb

                wandb_items = []
                for log_item in video_logs:
                    wandb_items.append(
                        wandb.Video(log_item["videos"], caption=log_item["validation_prompt"], fps=16, format="mp4")
                    )

                # for validation_prompt, video in formatted_images:
                #     wandb_images.append(wandb.Image(image, caption=validation_prompt, file_type="jpg"))
                tracker.log({"validation": wandb_items})
            else:
                logger.warn(f"image logging not implemented for {tracker.name}")

    def concatenate_videos(video_data, videos_per_row=3, video_format="mp4"):
        videos = [torch.from_numpy(log["videos"]).to(torch.uint8) for log in video_data]  # T,C,H,W

        # Calculate grid dimensions
        num_videos = len(videos)
        num_rows = (num_videos + videos_per_row - 1) // videos_per_row

        # Get height and width of each video
        num_frames, num_channels, height, width = videos[0].shape

        # Calculate total grid dimensions
        total_width = width * min(videos_per_row, num_videos)
        total_height = height * num_rows

        # Create output tensor
        grid_video = torch.zeros((num_frames, num_channels, total_height, total_width), dtype=videos[0].dtype)

        # Process each video
        for i, video in enumerate(videos):

            # Calculate position in grid
            row = i // videos_per_row
            col = i % videos_per_row

            # Calculate offsets
            y_offset = row * height
            x_offset = col * width

            # Get dimensions
            h, w = video.shape[2:]

            # Place frame in grid
            grid_video[:, :, y_offset : y_offset + h, x_offset : x_offset + w] = video

        return grid_video

    if config.train.local_save_vis and accelerator.is_main_process:
        file_format = "mp4"
        local_vis_save_path = osp.join(config.work_dir, "log_vis")
        os.umask(0o000)
        os.makedirs(local_vis_save_path, exist_ok=True)
        num_videos = len(video_logs) if init_noise is None else len(video_logs) // 2

        # samples without init noise
        concatenated_video = concatenate_videos(
            video_logs[:num_videos], videos_per_row=num_videos, video_format=file_format
        )
        save_path = osp.join(local_vis_save_path, f"vis_{step}.{file_format}")
        # print(f"Save for no init {save_path}, num videos : {num_videos}, concatenated_video shape {concatenated_video.shape}")
        save_video = concatenated_video.permute(0, 2, 3, 1)
        # write_video(save_path, save_video, fps=16, video_codec="libx264")
        # write video
        writer = imageio.get_writer(save_path, fps=16, codec="libx264", quality=8)
        for frame in save_video.numpy():
            writer.append_data(frame)
        writer.close()

        if init_noise is not None:
            concatenated_video = concatenate_videos(
                video_logs[num_videos:], videos_per_row=num_videos, video_format=file_format
            )
            save_path = osp.join(local_vis_save_path, f"vis_{step}_w_init.{file_format}")
            # print(f"Save for init noise {save_path}")
            save_video = concatenated_video.permute(0, 2, 3, 1)
            # write_video(save_path, save_video, fps=16, video_codec="libx264")
            # write video
            writer = imageio.get_writer(save_path, fps=16, codec="libx264", quality=8)
            for frame in save_video.numpy():
                writer.append_data(frame)
            writer.close()

    model.train()
    del vae
    flush()
    return video_logs


def train(
    config, args, accelerator, model, model_ema, optimizer, lr_scheduler, train_dataloader, train_diffusion, logger
):
    if getattr(config.train, "debug_nan", False):
        DebugUnderflowOverflow(model, max_frames_to_save=100)
        logger.info("NaN debugger registered. Start to detect overflow during training.")
    log_buffer = LogBuffer()

    if config.train.offload_vae:
        vae.to("cpu")
    if config.train.offload_text_encoder:
        text_encoder.to("cpu")

    null_y = torch.load(null_embed_path, map_location="cpu")
    null_y = null_y["uncond_prompt_embeds"].to(accelerator.device)

    global_step = start_step + 1
    skip_step = max(config.train.skip_step, global_step) % train_dataloader_len
    skip_step = skip_step if skip_step < (train_dataloader_len - 20) else 0
    loss_nan_timer = 0

    # Cache Dataset for BatchSampler
    if args.caching and config.model.multi_scale:
        caching_start = time.time()
        logger.info(
            f"Start caching your dataset for batch_sampler at {cache_file}. \n"
            f"This may take a lot of time...No training will launch"
        )
        train_dataloader.batch_sampler.sampler.set_start(max(train_dataloader.batch_sampler.exist_ids, 0))
        accelerator.wait_for_everyone()
        for index, _ in enumerate(train_dataloader):
            accelerator.wait_for_everyone()
            if index % 2000 == 0:
                logger.info(
                    f"rank: {rank}, Cached file len: {len(train_dataloader.batch_sampler.cached_idx)} / {len(train_dataloader)}"
                )
                print(
                    f"rank: {rank}, Cached file len: {len(train_dataloader.batch_sampler.cached_idx)} / {len(train_dataloader)}"
                )
            if (time.time() - caching_start) / 3600 > 3.7:
                json.dump(train_dataloader.batch_sampler.cached_idx, open(cache_file, "w"), indent=4)
                accelerator.wait_for_everyone()
                break
            if len(train_dataloader.batch_sampler.cached_idx) == len(train_dataloader) - 1000:
                logger.info(
                    f"Saving rank: {rank}, Cached file len: {len(train_dataloader.batch_sampler.cached_idx)} / {len(train_dataloader)}"
                )
                json.dump(train_dataloader.batch_sampler.cached_idx, open(cache_file, "w"), indent=4)
            accelerator.wait_for_everyone()
            continue
        accelerator.wait_for_everyone()
        print(f"Saving rank-{rank} Cached file len: {len(train_dataloader.batch_sampler.cached_idx)}")
        json.dump(train_dataloader.batch_sampler.cached_idx, open(cache_file, "w"), indent=4)
        return

    # Now you train the model
    for epoch in range(start_epoch + 1, config.train.num_epochs + 1):
        time_start, last_tic = time.time(), time.time()
        sampler = (
            train_dataloader.batch_sampler.sampler
            if (num_replicas > 1 or config.model.multi_scale)
            else train_dataloader.sampler
        )
        sampler.set_epoch(epoch)
        sampler.set_start(max((skip_step - 1) * config.train.train_batch_size, 0))
        if skip_step > 1 and accelerator.is_main_process:
            logger.info(f"Skipped Steps: {skip_step}")
        skip_step = 1
        data_time_start = time.time()
        data_time_all = 0
        lm_time_all = 0
        vae_time_all = 0
        model_time_all = 0
        for step, batch in enumerate(train_dataloader):
            # one_logger_callbacks.on_train_batch_start()
            # image, json_info, key = batch
            accelerator.wait_for_everyone()
            data_time_all += time.time() - data_time_start
            vae_time_start = time.time()
            image_context = None
            image_embeds = None
            data_info = batch[3]
            if load_vae_feat:
                z = batch[0].to(accelerator.device)
                # print('z:', z.shape)
                if config.model.mhla_adjust:
                    z = z[..., :60, :100]
            else:
                if config.train.offload_vae:
                    vae.to(accelerator.device)

                with torch.no_grad():

                    # batch[0] shape: B,F,C,H,W
                    z = vae_encode(
                        config.vae.vae_type,
                        vae,
                        batch[0].permute(0, 2, 1, 3, 4).to(vae_dtype),
                        device=accelerator.device,
                        cache_key=data_info["cache_key"],
                        if_cache=config.vae.if_cache,
                    )  # B,F,C,H,W -> B,C,F,H,W
                    torch.cuda.empty_cache()

                    if config.task == "ti2v":

                        with torch.no_grad():
                            image_embeds = encode_image(
                                config.image_encoder.image_encoder_type,
                                image_encoder,
                                batch[0][:, 0],
                                device=accelerator.device,
                            )

                        if config.model.image_latent_mode == "repeat" or config.model.image_latent_mode == "zero":
                            image_context = vae_encode(
                                config.vae.vae_type,
                                vae,
                                batch[0][:, :1].permute(0, 2, 1, 3, 4).to(vae_dtype),
                                device=accelerator.device,
                            )  # B,1,C,H,W -> B,C,1,H,W -> B,C,1,H,W
                            num_latent_frames = z.shape[2]
                            if config.model.image_latent_mode == "repeat":
                                image_context = image_context.repeat(1, 1, num_latent_frames, 1, 1)  # B,C,F,H,W
                            elif config.model.image_latent_mode == "zero":
                                zero_context = torch.zeros_like(z)
                                length_image_context = image_context.shape[2]
                                zero_context[:, :, :length_image_context, :, :] = image_context
                                image_context = zero_context
                        elif config.model.image_latent_mode == "video_zero":
                            pad_video = torch.zeros_like(batch[0])
                            pad_video[:, :1] = batch[0][:, :1]
                            image_context = vae_encode(
                                config.vae.vae_type,
                                vae,
                                pad_video.permute(0, 2, 1, 3, 4).to(vae_dtype),
                                device=accelerator.device,
                            )
                        # frame index mask for WanI2V 14B
                        if config.model.mask == "first":

                            image_context_height, image_context_width = image_context.shape[3], image_context.shape[4]
                            msk = torch.ones(
                                image_context.size(0),
                                config.model.num_frames,
                                image_context_height,
                                image_context_width,
                                device=accelerator.device,
                            )
                            msk[:, 1:] = 0
                            if config.vae.vae_type == "WanVAE":
                                msk = torch.concat(
                                    [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
                                )  # 1,84,h,w
                                # print(f"msk.shape: {msk.shape}")

                            msk = msk.view(
                                image_context.size(0), msk.shape[1] // 4, 4, image_context_height, image_context_width
                            )  # 1,21,4,h,w
                            msk = msk.transpose(1, 2)  # 1,4,21,h,w
                            image_context = torch.cat([msk, image_context], dim=1)  # 1,C+4,f,h,w

                    if config.debug and step % 5 == 0 and accelerator.is_main_process:
                        reconstructed = vae_decode(config.vae.vae_type, vae, z.to(vae_dtype))
                        if isinstance(reconstructed, list):
                            reconstructed = torch.stack(reconstructed, dim=0)
                        reconstructed = reconstructed * 0.5 + 0.5
                        video = reconstructed[0].permute(1, 2, 3, 0).clamp(0, 1) * 255  # C,T,H,W -> T,H,W,C
                        video = video.to("cpu", dtype=torch.uint8).numpy()

                        os.makedirs(f"{config.work_dir}/log_vis", exist_ok=True)
                        imageio.mimwrite(
                            f"{config.work_dir}/log_vis/debug_recon_{step}.mp4",
                            video,
                            fps=16,
                            macro_block_size=None,
                            quality=8,
                        )
                        # save original video
                        # F,C,H,W -> F,H,W,C
                        video_original = (batch[0][0].permute(0, 2, 3, 1) * 0.5 + 0.5).clamp(0, 1) * 255
                        video_original = video_original.to("cpu", dtype=torch.uint8).numpy()
                        imageio.mimwrite(
                            f"{config.work_dir}/log_vis/debug_original_{step}.mp4",
                            video_original,
                            fps=16,
                            macro_block_size=None,
                            quality=8,
                        )
                        # imageio.mimwrite(f"{config.work_dir}/log_vis/debug_recon_{step}v2.mp4", v2, fps=16, macro_block_size=None, quality=8)
                        if config.task == "ti2v":
                            # decode and save the image context
                            image_context_save = image_context[:, 4:] if config.model.mask == "first" else image_context
                            reconstructed = vae_decode(config.vae.vae_type, vae, image_context_save.to(vae_dtype))
                            if isinstance(reconstructed, list):
                                reconstructed = torch.stack(reconstructed, dim=0)
                            reconstructed = reconstructed * 0.5 + 0.5
                            video_image_context = (
                                reconstructed[0].permute(1, 2, 3, 0).clamp(0, 1) * 255
                            )  # C,T,H,W -> T,H,W,C
                            video_image_context = video_image_context.to("cpu", dtype=torch.uint8).numpy()

                            imageio.mimwrite(
                                f"{config.work_dir}/log_vis/debug_image_context_{step}.mp4",
                                video_image_context,
                                fps=16,
                                macro_block_size=None,
                                quality=8,
                            )

                if config.train.offload_vae:
                    vae.to("cpu")
            accelerator.wait_for_everyone()
            vae_time_all += time.time() - vae_time_start
            clean_images = z

            lm_time_start = time.time()
            if load_text_feat:
                y = batch[1]  # bs, 1, N, C
                y_mask = batch[2]  # bs, 1, 1, N
            else:
                if config.train.offload_text_encoder:
                    text_encoder.to(accelerator.device)
                y = text_encoder(batch[1])
                if config.train.offload_text_encoder:
                    text_encoder.to("cpu")
                rand_null = torch.rand(len(y)) < config.model.class_dropout_prob
                y = [yi if not r else null_y.to(y[0].device, y[0].dtype) for yi, r in zip(y, rand_null)]

            if config.task == "ti2v":
                rand_null_image_context = torch.rand(image_context.size(0)) < config.model.class_dropout_prob
                image_context[rand_null_image_context] = 0

            # Sample a random timestep for each image
            bs = clean_images.shape[0]
            timesteps = torch.randint(
                0, config.scheduler.train_sampling_steps, (bs,), device=clean_images.device
            ).long()
            if config.scheduler.weighting_scheme in ["logit_normal"]:
                # adapting from diffusers.training_utils
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=config.scheduler.weighting_scheme,
                    batch_size=bs,
                    logit_mean=config.scheduler.logit_mean,
                    logit_std=config.scheduler.logit_std,
                    mode_scale=None,  # not used
                )
                timesteps = (u * config.scheduler.train_sampling_steps).long().to(clean_images.device)
            grad_norm = None
            accelerator.wait_for_everyone()
            lm_time_all += time.time() - lm_time_start
            model_time_start = time.time()

            # for multi-scale training, the seq length is different for each batch
            # clean_images : B,C,F,H,W
            video_pos_seq_len = math.ceil(
                (clean_images.size(3) * clean_images.size(4))
                / (config.model.patch_size[1] * config.model.patch_size[2])
                * clean_images.size(2)
                / config.model.patch_size[0]
            )
            if isinstance(block_mask, dict):
                cur_block_mask = block_mask[f"{clean_images.size(2)}x{clean_images.size(3)}x{clean_images.size(4)}"]
            else:
                cur_block_mask = block_mask
            if cur_block_mask is not None:
                cur_block_mask = cur_block_mask.to(accelerator.device)

            with accelerator.accumulate(model):
                # Predict the noise residual
                optimizer.zero_grad()
                loss_term = train_diffusion.training_losses(
                    model,
                    clean_images,
                    timesteps,
                    model_kwargs=dict(
                        context=y,
                        seq_len=video_pos_seq_len,
                        y=image_context,
                        clip_fea=image_embeds,
                        block_mask=cur_block_mask,
                    ),
                )
                loss = loss_term["loss"].mean()
                loss_all = loss

                if config.distill is not None:
                    # distill_loss = 0

                    with torch.no_grad():
                        # get teacher model output
                        teacher_model_output = train_diffusion.training_losses(
                            teacher_model,
                            clean_images,
                            timesteps,
                            model_kwargs=dict(
                                context=y, seq_len=video_pos_seq_len, y=image_context, clip_fea=image_embeds
                            ),
                        )
                    teacher_model_output = teacher_model_output["output"].float().detach()
                    model_output = loss_term["output"].float()
                    # compute distillation loss
                    distill_logit_loss = (
                        F.mse_loss(model_output, teacher_model_output) * config.distill.distill_logit_weight
                    )

                    # compute distillation loss
                    try:
                        teacher_attn_outputs = teacher_model.get_attn_output()
                    except:
                        # FSDP mode
                        teacher_attn_outputs = teacher_model.module.get_attn_output()

                    try:
                        model_attn_outputs = model.get_attn_output()
                    except:
                        # FSDP mode
                        model_attn_outputs = model.module.get_attn_output()

                    distill_attn_loss = 0
                    # specify the layer when register attn hook
                    for layer_idx, model_attn_output in model_attn_outputs.items():
                        teacher_attn_output = teacher_attn_outputs[layer_idx].detach().to(model_attn_output.device)
                        distill_attn_loss += F.mse_loss(model_attn_output.float(), teacher_attn_output.float())
                    distill_attn_loss = distill_attn_loss / len(model_attn_outputs) * config.distill.distill_attn_weight
                    # accelerator.backward(distill_attn_loss)
                    # distill_loss += distill_attn_loss

                    loss_all = loss + distill_attn_loss + distill_logit_loss

                accelerator.backward(loss_all)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.train.gradient_clip)
                    if not config.train.use_fsdp and config.train.ema_update and model_ema is not None:
                        ema_update(model_ema, model, config.train.ema_rate)

                optimizer.step()
                lr_scheduler.step()
                accelerator.wait_for_everyone()
                model_time_all += time.time() - model_time_start
            # one_logger_callbacks.on_train_batch_end()

            if torch.any(torch.isnan(loss)):
                loss_nan_timer += 1
            lr = lr_scheduler.get_last_lr()[0]
            logs = {args.loss_report_name: accelerator.gather(loss).mean().item()}
            if grad_norm is not None:
                # NOTE the grad norm cannot be synced when offload cpu
                logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())

            if config.distill is not None:
                logs.update(
                    distill_logit_loss=accelerator.gather(distill_logit_loss).mean().item(),
                    distill_attn_loss=accelerator.gather(distill_attn_loss).mean().item(),
                )

            log_buffer.update(logs)
            if (step + 1) % config.train.log_interval == 0 or (step + 1) == 1:
                accelerator.wait_for_everyone()
                t = (time.time() - last_tic) / config.train.log_interval
                t_d = data_time_all / config.train.log_interval
                t_m = model_time_all / config.train.log_interval
                t_lm = lm_time_all / config.train.log_interval
                t_vae = vae_time_all / config.train.log_interval
                avg_time = (time.time() - time_start) / (step + 1)
                eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - global_step - 1))))
                eta_epoch = str(
                    datetime.timedelta(
                        seconds=int(
                            avg_time
                            * (train_dataloader_len - sampler.step_start // config.train.train_batch_size - step - 1)
                        )
                    )
                )
                log_buffer.average()

                current_step = (
                    global_step - sampler.step_start // config.train.train_batch_size
                ) % train_dataloader_len
                current_step = train_dataloader_len if current_step == 0 else current_step

                info = (
                    f"Epoch: {epoch} | Global Step: {global_step} | Local Step: {current_step} // {train_dataloader_len}, "
                    f"total_eta: {eta}, epoch_eta:{eta_epoch}, time: all:{t:.3f}, model:{t_m:.3f}, data:{t_d:.3f}, "
                    f"lm:{t_lm:.3f}, vae:{t_vae:.3f}, lr:{lr:.3e}, Cap: {batch[5][0]}, "
                )

                info += ", ".join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
                last_tic = time.time()
                log_buffer.clear()
                data_time_all = 0
                model_time_all = 0
                lm_time_all = 0
                vae_time_all = 0
                if accelerator.is_main_process:
                    logger.info(info)

            logs.update(lr=lr)
            if accelerator.is_main_process:
                accelerator.log(logs, step=global_step)

            global_step += 1

            if loss_nan_timer > 20:
                raise ValueError("Loss is NaN too much times. Break here.")
            if (
                global_step % config.train.save_model_steps == 0
                or (time.time() - training_start_time) / 3600 > config.train.early_stop_hours
            ):
                torch.cuda.synchronize()
                accelerator.wait_for_everyone()
                # one_logger_callbacks.on_save_checkpoint_start(global_step=global_step)

                # Choose different saving methods based on whether FSDP is used
                if config.train.use_fsdp:
                    # FSDP mode
                    os.umask(0o000)
                    ckpt_saved_path = save_checkpoint(
                        work_dir=osp.join(config.work_dir, "checkpoints"),
                        epoch=epoch,
                        model=model,
                        accelerator=accelerator,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        step=global_step,
                        add_symlink=True,
                    )
                else:
                    # DDP mode
                    if accelerator.is_main_process:
                        os.umask(0o000)
                        ckpt_saved_path = save_checkpoint(
                            work_dir=osp.join(config.work_dir, "checkpoints"),
                            epoch=epoch,
                            step=global_step,
                            model=accelerator.unwrap_model(model),
                            model_ema=accelerator.unwrap_model(model_ema) if model_ema is not None else None,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            generator=generator,
                            add_symlink=True,
                        )
                # one_logger_callbacks.on_save_checkpoint_success(global_step=global_step)
                # one_logger_callbacks.on_save_checkpoint_end(global_step=global_step)

                if accelerator.is_main_process:
                    if config.train.online_metric and global_step % config.train.eval_metric_step == 0 and step > 1:
                        online_metric_monitor_dir = osp.join(config.work_dir, config.train.online_metric_dir)
                        os.makedirs(online_metric_monitor_dir, exist_ok=True)
                        with open(f"{online_metric_monitor_dir}/{ckpt_saved_path.split('/')[-1]}.txt", "w") as f:
                            f.write(osp.join(config.work_dir, "config.py") + "\n")
                            f.write(ckpt_saved_path)

                if (time.time() - training_start_time) / 3600 > config.train.early_stop_hours:
                    logger.info(f"Stopping training at epoch {epoch}, step {global_step} due to time limit.")
                    return

            if config.train.visualize and (global_step % config.train.eval_sampling_steps == 0 or (step + 1) == 1):
                if config.train.use_fsdp:
                    merged_state_dict = accelerator.get_state_dict(model)

                accelerator.wait_for_everyone()
                # one_logger_callbacks.on_validation_start()
                if (
                    config.train.fsdp_inference
                ):  # if we use super large model, we may need to fsdp to warp the inference model and not in the main process
                    # model_instance = model # NOTE we cannot change model instance here, it will regard the model instance as the local variable instead of the global variable
                    log_validation(
                        accelerator=accelerator,
                        config=config,
                        model=model.eval(),
                        logger=logger,
                        step=global_step,
                        device=accelerator.device,
                        vae=vae,
                        init_noise=validation_noise,
                    )
                else:
                    if (
                        accelerator.is_main_process
                    ):  # if we use super large model, we may need to fsdp to warp the inference model and not in the main process
                        if config.train.use_fsdp:
                            model_instance.load_state_dict(merged_state_dict)
                            model_instance.to(accelerator.device)

                        log_validation(
                            accelerator=accelerator,
                            config=config,
                            model=model_instance.eval(),
                            logger=logger,
                            step=global_step,
                            device=accelerator.device,
                            vae=vae,
                            init_noise=validation_noise,
                        )
                # one_logger_callbacks.on_validation_end()

            model.train()
            # avoid dead-lock of multiscale data batch sampler
            if (
                config.model.multi_scale
                and (train_dataloader_len - sampler.step_start // config.train.train_batch_size - step) < 30
            ):
                global_step = (
                    (global_step + train_dataloader_len - 1) // train_dataloader_len
                ) * train_dataloader_len + 1
                logger.info("Early stop current iteration")
                skip_first_batches(train_dataloader, True)
                break

            data_time_start = time.time()

        if epoch % config.train.save_model_epochs == 0 or epoch == config.train.num_epochs and not config.debug:
            accelerator.wait_for_everyone()
            torch.cuda.synchronize()

            # one_logger_callbacks.on_save_checkpoint_start(global_step=global_step)
            # Choose different saving methods based on whether FSDP is used
            if config.train.use_fsdp:
                # FSDP mode
                os.umask(0o000)
                ckpt_saved_path = save_checkpoint(
                    work_dir=osp.join(config.work_dir, "checkpoints"),
                    epoch=epoch,
                    model=model,
                    accelerator=accelerator,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    step=global_step,
                    add_symlink=True,
                )
            else:
                # DDP mode
                if accelerator.is_main_process:
                    os.umask(0o000)
                    ckpt_saved_path = save_checkpoint(
                        osp.join(config.work_dir, "checkpoints"),
                        epoch=epoch,
                        step=global_step,
                        model=accelerator.unwrap_model(model),
                        model_ema=accelerator.unwrap_model(model_ema) if model_ema is not None else None,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        generator=generator,
                        add_symlink=True,
                    )

            # one_logger_callbacks.on_save_checkpoint_success(global_step=global_step)
            # one_logger_callbacks.on_save_checkpoint_end(global_step=global_step)

            if accelerator.is_main_process:
                online_metric_monitor_dir = osp.join(config.work_dir, config.train.online_metric_dir)
                os.makedirs(online_metric_monitor_dir, exist_ok=True)
                with open(f"{online_metric_monitor_dir}/{ckpt_saved_path.split('/')[-1]}.txt", "w") as f:
                    f.write(osp.join(config.work_dir, "config.py") + "\n")
                    f.write(ckpt_saved_path)


@pyrallis.wrap()
def main(cfg: WanConfig) -> None:
    global train_dataloader_len, start_epoch, start_step, vae, generator, num_replicas, rank, training_start_time, image_encoder
    global load_vae_feat, load_text_feat, validation_noise, text_encoder, tokenizer
    global max_length, validation_prompts, latent_height, latent_width, latent_temp, valid_prompt_embed_suffix, null_embed_path
    global video_width, video_height, cache_file, total_steps, vae_dtype, model_instance, teacher_model
    global block_mask

    # Load environment variables from .env file
    load_dotenv()
    
    # Set wandb API key from environment variable
    wandb_api_key = os.getenv('WANDB_API_KEY')
    if wandb_api_key:
        os.environ['WANDB_API_KEY'] = wandb_api_key

    config = cfg
    args = cfg

    # Record app start time at the beginning of main
    app_start_time = int(time.time() * 1000)

    # 1.Initialize training mode
    if config.train.use_fsdp:
        set_fsdp_env()
        init_train = "FSDP"
    else:
        init_train = "DDP"

    training_start_time = time.time()
    load_from = True

    if args.resume_from or config.model.resume_from:
        load_from = False
        config.model.resume_from = dict(
            checkpoint=args.resume_from or config.model.resume_from,
            load_ema=False,
            resume_optimizer=True,
            resume_lr_scheduler=config.train.resume_lr_scheduler,
        )

    if args.debug:
        config.train.log_interval = 1
        config.train.train_batch_size = min(64, config.train.train_batch_size)
        if config.train.use_fsdp:
            os.environ["FSDP_SHARDING_STRATEGY"] = "FULL_SHARD"
        # args.report_to = "tensorboard"

    os.umask(0o000)
    os.makedirs(config.work_dir, exist_ok=True)

    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug

    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.model.mixed_precision,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=osp.join(config.work_dir, "logs"),
        kwargs_handlers=[init_handler],
    )

    log_name = "train_log.log"
    logger = get_root_logger(osp.join(config.work_dir, log_name))
    logger.info(accelerator.state)

    config.train.seed = init_random_seed(getattr(config.train, "seed", None))
    set_random_seed(config.train.seed + int(os.environ["LOCAL_RANK"]))
    generator = torch.Generator(device="cpu").manual_seed(config.train.seed)

    if accelerator.is_main_process:
        pyrallis.dump(config, open(osp.join(config.work_dir, "config.yaml"), "w"), sort_keys=False, indent=4)
        if args.report_to == "wandb":
            import wandb

            wandb.init(project=args.tracker_project_name, name=args.name, resume="allow", id=args.name)

    cluster = os.environ.get("CLUSTER", "cs")
    if cluster == "cs":
        config.train.early_stop_hours = min(3.85, config.train.early_stop_hours)
    elif cluster == "nrt":
        config.train.early_stop_hours = min(1.85, config.train.early_stop_hours)

    logger.info(f"Config: \n{config}")
    logger.info(f"World_size: {get_world_size()}, seed: {config.train.seed}")
    logger.info(f"Initializing: {init_train} for training")

    # LoRA configuration validation
    use_lora = hasattr(config, "lora") and config.lora is not None and config.lora.use_lora
    if use_lora:
        logger.info("LoRA configuration detected - enabling LoRA training")
        logger.info(f"LoRA rank: {config.lora.rank}")
        logger.info(f"LoRA alpha: {config.lora.alpha}")
        logger.info(f"LoRA target modules: {config.lora.target_modules}")
        logger.info(f"LoRA dropout: {config.lora.dropout}")

    else:
        logger.info("No LoRA configuration found - using standard training")

    # init one logger callback
    config.train_iterations_target = 70000 * 10
    config.model_max_length = 30 * 52 * 21
    config.global_world_size = get_world_size()
    # one_logger_callbacks = OneLoggerUtils(one_logger_callback_config(config))
    # one_logger_callbacks.on_app_start(app_start_time=app_start_time)

    # 2. build dataloader
    config.data.data_dir = (
        config.data.data_dir if isinstance(config.data.data_dir, dict) else {"default": config.data.data_dir}
    )
    config.data.data_dir = {
        k: data if data.startswith(("https://", "http://", "gs://", "/", "~")) else osp.abspath(osp.expanduser(data))
        for k, data in config.data.data_dir.items()
    }

    # one_logger_callbacks.on_dataloader_init_start()

    num_replicas = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    num_frames = config.model.num_frames
    config.data.num_frames = num_frames
    max_length = config.text_encoder.text_len
    # video dataset
    if config.model.aspect_ratio_type is not None:
        config.data.aspect_ratio_type = config.model.aspect_ratio_type

    # set a random seed
    random_seed = init_random_seed(None)
    set_random_seed(random_seed + int(os.environ["LOCAL_RANK"]))

    dataset = build_dataset(
        asdict(config.data),
        resolution=None,
        aspect_ratio_type=config.model.aspect_ratio_type,
        max_length=max_length,
        config=config,
        caption_proportion=config.data.caption_proportion,
        sort_dataset=config.data.sort_dataset,
        vae_downsample_rate=config.vae.vae_stride[-1],
        num_frames=num_frames,
    )

    # aspect_ratio_key = random.choice(list(dataset.aspect_ratio.keys()))
    aspect_ratio_key = "0.57"
    video_height, video_width = map(int, dataset.aspect_ratio[aspect_ratio_key])
    # video_height, video_width = int(video_height), int(video_width)
    latent_width = int(video_width) // config.vae.vae_stride[2]
    if config.model.mhla_adjust:
        latent_width = 100
    latent_height = int(video_height) // config.vae.vae_stride[1]
    latent_temp = int(config.model.num_frames - 1) // config.vae.vae_stride[0] + 1

    validation_noise = (
        torch.randn(
            1, config.vae.vae_latent_dim, latent_temp, latent_height, latent_width, device="cpu", generator=generator
        )
        if getattr(config.train, "deterministic_validation", False)
        else None
    )

    # get the max seq len across aspect ratios
    max_area, max_aspect_ratio = 0, 0.57
    for aspect_ratio in dataset.aspect_ratio:
        area = dataset.aspect_ratio[aspect_ratio][0] * dataset.aspect_ratio[aspect_ratio][1]
        if area > max_area:
            max_area = area
            max_aspect_ratio = aspect_ratio
    max_area_height, max_area_width = map(int, dataset.aspect_ratio[max_aspect_ratio])
    # video_pos_seq_len = math.ceil(
    #     (max_area_height // config.vae.vae_stride[1] * max_area_width // config.vae.vae_stride[2]) / (config.model.patch_size[1] * config.model.patch_size[2]) * latent_temp
    # )

    accelerator.wait_for_everyone()
    sampler = DistributedRangedSampler(dataset, num_replicas=num_replicas, rank=rank)
    if config.model.multi_scale:
        batch_sampler = AspectRatioBatchSamplerVideo(
            sampler=sampler,
            dataset=dataset,
            batch_size=config.train.train_batch_size,
            aspect_ratios=dataset.aspect_ratio,
            drop_last=True,
            ratio_nums=dataset.ratio_nums,
            config=config,
            valid_num=config.data.valid_num,
        )
        train_dataloader = build_dataloader(dataset, batch_sampler=batch_sampler, num_workers=config.train.num_workers)

    else:
        train_dataloader = build_dataloader(
            dataset,
            num_workers=config.train.num_workers,
            batch_size=config.train.train_batch_size,
            shuffle=False,
            sampler=sampler,
        )
    train_dataloader_len = len(train_dataloader)
    # one_logger_callbacks.on_dataloader_init_end()

    load_vae_feat = getattr(train_dataloader.dataset, "load_vae_feat", False)
    load_text_feat = getattr(train_dataloader.dataset, "load_text_feat", False)

    # 3. Build VAE, Text Encoder, Image Encoder

    # VAE
    vae_dtype = get_weight_dtype(config.vae.weight_dtype)
    vae = get_vae(
        config.vae.vae_type, config.vae.vae_pretrained, accelerator.device, dtype=vae_dtype, config=config.vae
    )

    if not config.data.load_vae_feat and config.vae.cache_dir is not None:
        vae_cache_dir = os.path.join(
            config.vae.cache_dir,
            f"{config.vae.vae_type}_{num_frames}x{video_height}x{video_width}",
        )
        os.makedirs(vae_cache_dir, exist_ok=True)
        vae.cfg.cache_dir = vae_cache_dir
        logger.info(f"Cache VAE latent of {num_frames}x{video_height}x{video_width} to {vae_cache_dir}")

    text_encoder = T5EncoderModel(
        text_len=config.text_encoder.text_len,
        dtype=get_weight_dtype(config.text_encoder.t5_dtype),
        device=accelerator.device,
        checkpoint_path=config.text_encoder.t5_checkpoint,
        tokenizer_path=config.text_encoder.t5_tokenizer,
    )
    image_encoder = None
    if config.image_encoder.image_encoder_type is not None:
        image_encoder = get_image_encoder(
            config.image_encoder.image_encoder_type,
            config.image_encoder.image_encoder_pretrained,
            config.image_encoder.image_encoder_tokenizer,
            accelerator.device,
            dtype=get_weight_dtype(config.image_encoder.weight_dtype),
            config=config.image_encoder,
        )

    # hard code here
    text_embed_dim = 4096
    os.makedirs(config.train.null_embed_root, exist_ok=True)
    null_embed_path = osp.join(
        config.train.null_embed_root,
        f"null_embed_diffusers_{config.text_encoder.t5_model}_{max_length}token_{text_embed_dim}.pth",
    )

    # 4. Preparing embeddings for visualization. We put it here for saving GPU memory
    if config.train.visualize and len(config.train.validation_prompts):
        valid_prompt_embed_suffix = (
            f"{max_length}token_{config.text_encoder.t5_model}_{text_embed_dim}.pth"
            if not config.task == "ti2v"
            else f"{max_length}token_{config.text_encoder.t5_model}_{text_embed_dim}_{config.model.num_frames}x{config.model.video_height}x{config.model.video_width}.pth"
        )
        validation_prompts = config.train.validation_prompts
        skip = True
        uuid_sys_prompt = hashlib.sha256(b"").hexdigest()
        if config.task == "ti2v":
            uuid_sys_prompt += f"_{config.model.num_frames}x{config.model.video_height}x{config.model.video_width}_{config.vae.vae_type}_{config.model.image_latent_mode}"
            if config.image_encoder.image_encoder_type is not None:
                uuid_sys_prompt += f"_{config.image_encoder.image_encoder_type}"
        config.train.valid_prompt_embed_root = osp.join(config.train.valid_prompt_embed_root, uuid_sys_prompt)
        Path(config.train.valid_prompt_embed_root).mkdir(parents=True, exist_ok=True)
        for prompt in validation_prompts:
            prompt_embed_path = osp.join(
                config.train.valid_prompt_embed_root, f"{prompt[:50]}_{valid_prompt_embed_suffix}"
            )
            if not (osp.exists(prompt_embed_path) and osp.exists(null_embed_path)):
                skip = False
                logger.info("Preparing Visualization prompt embeddings...")
                break

        if accelerator.is_main_process and not skip:

            for prompt_idx, prompt in enumerate(validation_prompts):
                prompt_embed_path = osp.join(
                    config.train.valid_prompt_embed_root, f"{prompt[:50]}_{valid_prompt_embed_suffix}"
                )
                caption_emb = text_encoder(prompt)[0]
                # text_encoder.to("cpu")

                save_dict = {"caption_embeds": caption_emb, "emb_mask": None}
                if config.task == "ti2v":

                    image = read_image_from_path(
                        config.train.validation_images[prompt_idx],
                        (config.model.video_height, config.model.video_width),
                    )  # C,H,W
                    with torch.no_grad():
                        image_embeds = encode_image(
                            config.image_encoder.image_encoder_type,
                            image_encoder,
                            image[None],
                            device=accelerator.device,
                        )
                        if config.model.image_latent_mode == "repeat" or config.model.image_latent_mode == "zero":
                            image_context = vae_encode(
                                config.vae.vae_type, vae, image[None, :, None].to(vae_dtype), device=accelerator.device
                            )  # 1,C,1,H,W

                        elif config.model.image_latent_mode == "video_zero":
                            torch.cuda.empty_cache()
                            dummy_vid = torch.zeros_like(image[None, :, None]).repeat(
                                1, 1, config.model.num_frames, 1, 1
                            )  # C,H,W -> 1,C,1,H,W -> 1,C,F,H,W
                            dummy_vid[:, :, :1] = image[None, :, None]
                            image_context = vae_encode(
                                config.vae.vae_type, vae, dummy_vid.to(vae_dtype), device=accelerator.device
                            )  # 1,C,F,H,W

                    logger.info(
                        f"image latent mode {config.model.image_latent_mode} image_context shape {image_context.shape}"
                    )
                    save_dict["image_context"] = image_context.cpu()
                    if image_embeds is not None:
                        save_dict["image_embeds"] = image_embeds.cpu()

                torch.save(
                    save_dict,
                    prompt_embed_path,
                )

            null_token_emb = text_encoder("")[0]
            torch.save(
                {"uncond_prompt_embeds": null_token_emb, "uncond_prompt_embeds_mask": None},
                null_embed_path,
            )
            if config.data.load_text_feat:
                del tokenizer
                del text_encoder
            del null_token_emb
            flush()

    # TODO 5. build scheduler

    pred_sigma = getattr(config.scheduler, "pred_sigma", True)
    learn_sigma = getattr(config.scheduler, "learn_sigma", True) and pred_sigma

    train_diffusion = Scheduler(
        str(config.scheduler.train_sampling_steps),
        noise_schedule=config.scheduler.noise_schedule,
        predict_flow_v=config.scheduler.predict_flow_v,
        learn_sigma=learn_sigma,
        pred_sigma=pred_sigma,
        snr=config.train.snr_loss,
        flow_shift=config.scheduler.flow_shift,
    )
    predict_info = (
        f"flow-prediction: {config.scheduler.predict_flow_v}, noise schedule: {config.scheduler.noise_schedule}"
    )
    if "flow" in config.scheduler.noise_schedule:
        predict_info += f", flow shift: {config.scheduler.flow_shift}"
    if config.scheduler.weighting_scheme in ["logit_normal", "mode"]:
        predict_info += (
            f", flow weighting: {config.scheduler.weighting_scheme}, "
            f"logit-mean: {config.scheduler.logit_mean}, logit-std: {config.scheduler.logit_std}"
        )
    logger.info(predict_info)

    # 6. build DiT models
    # one_logger_callbacks.on_model_init_start()
    logger.info("Start building model")
    model_build_start_time = time.time()
    
    if config.model.from_pretrained is None:
        model_configs = init_model_configs(config.model, config.vae)
        print('model_configs: ', model_configs, config.model)
        model = WanLinearAttentionModel(**model_configs).train()

    elif config.model.from_pretrained.endswith(".json"):
        model = WanModel.from_config(
            config.model.from_pretrained,
        ).train()
    elif os.path.isdir(config.model.from_pretrained) and os.path.exists(
        os.path.join(config.model.from_pretrained, "config.json")
    ):
        model = WanModel.from_pretrained(
            config.model.from_pretrained,
            device_map="cpu",
        ).train()

    model.gradient_checkpointing = config.train.grad_checkpointing
    model.lr_scale = config.train.lr_scale

    # one_logger_callbacks.on_model_init_end()
    model_build_end_time = time.time()
    logger.info(f"Finish building model, time cost: {model_build_end_time - model_build_start_time:.2f}s")

    # Apply LoRA if configured (improved approach: LoRA first, then load base weights)
    if use_lora:
        logger.info("Applying LoRA to model using improved approach...")

        # Step 1: Freeze all base model parameters
        for param in model.parameters():
            param.requires_grad = False

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

        # Apply LoRA to the model
        model = get_peft_model(model, peft_lora_config)

        # Step 3: Unfreeze additional trainable layers if specified
        if hasattr(config.lora, "additional_trainable_layers") and config.lora.additional_trainable_layers:
            for layer_name in config.lora.additional_trainable_layers:
                for name, module in model.named_modules():
                    if layer_name in name:
                        for param in module.parameters():
                            param.requires_grad = True
                        logger.info(f"Unfrozen additional trainable layer: {layer_name} in {name}")

        # Log parameter counts
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        lora_only_params = sum(param.numel() for param in get_peft_model_state_dict(model).values())

        logger.info("LoRA setup complete:")
        logger.info(f"  Total parameters: {total_params / 1e6:.2f}M")
        logger.info(f"  LoRA parameters: {lora_only_params / 1e6:.2f}M")
        logger.info(f"  Additional trainable: {(trainable_params - lora_only_params) / 1e6:.2f}M")
        logger.info(f"  Total trainable: {trainable_params / 1e6:.2f}M ({100 * trainable_params / total_params:.2f}%)")

    if (not config.train.use_fsdp) and config.train.ema_update:
        model_ema = deepcopy(model).eval()
        logger.info("Creating EMA model for DDP mode")
    elif config.train.use_fsdp and config.train.ema_update:
        logger.warning("EMA update is not supported in FSDP mode. Setting model_ema to None.")
        model_ema = None
    else:
        model_ema = None

    logger.info(
        colored(
            f"{model.__class__.__name__}:{config.model.model}, "
            f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M",
            "green",
            attrs=["bold"],
        )
    )

    if config.train.use_fsdp:
        if config.model.from_pretrained is None:
            model_instance = WanLinearAttentionModel.from_config(
                model.config,
            )
        else:
            model_instance = WanModel.from_config(
                model.config,
            )
    elif model_ema is not None:
        model_instance = deepcopy(model_ema)
    else:
        model_instance = model

    if use_lora and not isinstance(model_instance, PeftModel):

        # build model instance to be a PeftModel
        model_instance = get_peft_model(model_instance, peft_lora_config)

    # 4-1. load model
    if args.load_from is not None:
        config.model.load_from = args.load_from
    if config.model.load_from is not None and load_from:
        # one_logger_callbacks.on_load_checkpoint_start()
        _, missing, unexpected, _, _ = load_checkpoint(
            checkpoint=config.model.load_from,
            model=model,
            model_ema=model_ema,
            FSDP=config.train.use_fsdp,
            load_ema=config.model.resume_from.get("load_ema", False),
            null_embed_path=null_embed_path,
        )
        # one_logger_callbacks.on_load_checkpoint_end()
        logger.warning(f"Missing keys: {missing}")
        logger.warning(f"Unexpected keys: {unexpected}")

    if config.train.ema_update and not config.train.use_fsdp and model_ema is not None:
        ema_update(model_ema, model, 0.0)

    block_mask = None
    # Aura Attention Design # TODO: add a reference here
    if config.model.attn_mask == "diagonal":

        from matplotlib import pyplot as plt

        flex_attn_padding = 128
        os.makedirs(config.model.diagonal_mask_root, exist_ok=True)
        diagonal_mask_path = f"{config.model.diagonal_mask_root}/nlogn_mask_h{int(latent_height/model.patch_size[-2])}_w{int(latent_width/model.patch_size[-1])}_f{int(latent_temp/model.patch_size[0])}_p{flex_attn_padding}_b{config.model.diagonal_block_size}.pt"
        if not os.path.exists(diagonal_mask_path):
            if accelerator.is_main_process:
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
                plt.imshow(diagonal_mask.cpu().numpy()[:, :], cmap="hot")
                plt.colorbar()
                plt.title("diagonal mask")
                plt.savefig(diagonal_mask_path.replace(".pt", ".png"))
                plt.close()
                logger.info(
                    # f"Saving diagonal mask image to {config.work_dir}/nlogn_mask_h{int(latent_height/model.patch_size[-2])}_w{int(latent_width/model.patch_size[-1])}_f{int(latent_temp/model.patch_size[0])}_p{flex_attn_padding}_b{config.model.diagonal_block_size}.png"
                )
                logger.info(f"Saving diagonal mask to {diagonal_mask_path}")
                torch.save(diagonal_mask, diagonal_mask_path)
        else:
            logger.info(f"Loading diagonal mask from {diagonal_mask_path}")
            model.load_diagonal_mask(path=diagonal_mask_path)

        if accelerator.is_main_process:
            model_instance.load_diagonal_mask(path=diagonal_mask_path)
    elif config.model.attn_mask == "radial_sparse":

        # radial sparse attn, requires the number of tokens to be a multiple of 128
        from matplotlib import pyplot as plt

        from tools.attn_mask.gen_nlogn_mask import gen_log_mask_shrinked

        block_size = 128
        os.makedirs(config.model.diagonal_mask_root, exist_ok=True)
        block_mask = {}
        for aspect_ratio in dataset.aspect_ratio:
            vh, vw = map(int, dataset.aspect_ratio[aspect_ratio])
            lw = int(vw) // config.vae.vae_stride[2]
            lh = int(vh) // config.vae.vae_stride[1]
            lt = int(config.model.num_frames - 1) // config.vae.vae_stride[0] + 1
            diagonal_mask_path = f"{config.model.diagonal_mask_root}/nlogn_radial_mask_h{int(lh/model.patch_size[-2])}_w{int(lw/model.patch_size[-1])}_f{int(lt/model.patch_size[0])}.pt"
            if os.path.exists(diagonal_mask_path):
                logger.info(f"Loading radial sparse mask from {diagonal_mask_path}")
                cur_mask = torch.load(diagonal_mask_path, map_location="cpu")
                block_mask[f"{lt}x{lh}x{lw}"] = cur_mask
                continue

            logger.info(f"Creating radial sparse mask for {diagonal_mask_path}")
            num_token = int(lw / model.patch_size[-1]) * int(lh / model.patch_size[-2]) * int(lt / model.patch_size[0])
            assert num_token % block_size == 0, f"Number of tokens must be a multiple of {block_size}"
            num_frames = int(lt / model.patch_size[0])
            diagonal_mask = gen_log_mask_shrinked(num_token, num_token, num_frames, block_size).bool()
            block_mask[f"{lt}x{lh}x{lw}"] = diagonal_mask

            if accelerator.is_main_process:
                plt.imshow(diagonal_mask.cpu().numpy()[:, :], cmap="hot")
                plt.colorbar()
                plt.title("radial sparse mask")
                plt.savefig(diagonal_mask_path.replace(".pt", ".png"))
                plt.close()
                # logger.info(
                # f"Saving radial sparse mask image to {diagonal_mask_path.replace('.pt', '.png')}"
                # )
                logger.info(f"Saving radial sparse mask to {diagonal_mask_path}")
                torch.save(diagonal_mask, diagonal_mask_path)

        model.diagonal_mask = None
        model_instance.diagonal_mask = None

    block_mask = model_instance.diagonal_mask if block_mask is None else block_mask
    # 5. build distill model if needed
    teacher_model = None
    if config.distill is not None:
        teacher_model_configs = init_model_configs(config.distill.model, config.vae)
        teacher_model = WanLinearAttentionModel(**teacher_model_configs).eval()
        # both teacher and student model should have the same attention hook
        teacher_model.register_attn_hook(config.model.linear_attn_idx, device="cpu")
        model.register_attn_hook(config.model.linear_attn_idx, device=accelerator.device)

        # load teacher model
        if config.distill.model.load_model_ckpt:
            teacher_model.load_model_ckpt(config.distill.model.load_model_ckpt, init_patch_embedding=False)
        else:
            raise ValueError(f"Teacher model checkpoint not found: {config.distill.model.load_model_ckpt}")
        logger.info(
            colored(
                f"{teacher_model.__class__.__name__}:{config.distill.model.model}, "
                f"Teacher Model Parameters: {sum(p.numel() for p in teacher_model.parameters()) / 1e6:.2f}M",
                "green",
                attrs=["bold"],
            )
        )

        # freeze teacher model
        for param in teacher_model.parameters():
            param.requires_grad = False

    # 6. build optimizer and lr scheduler
    # If specify trainable modules, only train the specified modules
    if config.train.train_la_only:
        logger.info("Checking for WanLinearAttention modules...")
        # set all parameters to False
        for param in model.parameters():
            param.requires_grad = False

        for name, module in model.named_modules():
            if isinstance(module, WanLinearAttention):
                logger.info(f"Found WanLinearAttention module: {name}")
                for param in module.parameters():
                    param.requires_grad = True

    lr_scale_ratio = 1
    # one_logger_callbacks.on_optimizer_init_start()
    if getattr(config.train, "auto_lr", None):
        lr_scale_ratio = auto_scale_lr(
            config.train.train_batch_size * get_world_size() * config.train.gradient_accumulation_steps,
            config.train.optimizer,
            **config.train.auto_lr,
        )
    optimizer = build_optimizer(model, config.train.optimizer)
    # one_logger_callbacks.on_optimizer_init_end()
    if config.train.lr_schedule_args and config.train.lr_schedule_args.get("num_warmup_steps", None):
        config.train.lr_schedule_args["num_warmup_steps"] = (
            config.train.lr_schedule_args["num_warmup_steps"] * num_replicas
        )
    lr_scheduler = build_lr_scheduler(config.train, optimizer, train_dataloader, lr_scale_ratio)
    logger.warning(
        f"{colored('Basic Setting: ', 'green', attrs=['bold'])}"
        f"lr: {config.train.optimizer['lr']:.5f}, bs: {config.train.train_batch_size}, gc: {config.train.grad_checkpointing}, "
        f"gc_accum_step: {config.train.gradient_accumulation_steps}, qk norm: {config.model.qk_norm}, "
        f"fp32 attn: {config.model.fp32_attention}, "
        f"text encoder: {config.text_encoder.t5_model}, captions: {config.data.caption_proportion}, precision: {config.model.mixed_precision}"
    )

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        try:
            accelerator.init_trackers(args.tracker_project_name, tracker_config)
        except:
            accelerator.init_trackers(f"tb_{timestamp}")

    start_epoch = 0
    start_step = 0
    total_steps = train_dataloader_len * config.train.num_epochs

    # 7. Resume training
    logger.info("Start loading checkpoint")
    if config.model.resume_from is not None and config.model.resume_from["checkpoint"] is not None:
        rng_state = None
        ckpt_path = osp.join(config.work_dir, "checkpoints")
        check_flag = osp.exists(ckpt_path) and len(os.listdir(ckpt_path)) != 0

        if config.model.resume_from["checkpoint"] == "latest":
            if check_flag:
                config.model.resume_from["resume_optimizer"] = True
                config.model.resume_from["resume_lr_scheduler"] = True
                checkpoints = os.listdir(ckpt_path)
                if "latest.pth" in checkpoints and osp.exists(osp.join(ckpt_path, "latest.pth")):
                    config.model.resume_from["checkpoint"] = osp.realpath(osp.join(ckpt_path, "latest.pth"))
                else:
                    checkpoints = [i for i in checkpoints if i.startswith("epoch_")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.replace(".pth", "").split("_")[3]))
                    config.model.resume_from["checkpoint"] = osp.join(ckpt_path, checkpoints[-1])
            else:
                config.model.resume_from["resume_optimizer"] = config.train.load_from_optimizer
                config.model.resume_from["resume_lr_scheduler"] = config.train.load_from_lr_scheduler
                config.model.resume_from["checkpoint"] = config.model.load_from

        if config.model.resume_from["checkpoint"] is not None:
            # one_logger_callbacks.on_load_checkpoint_start()
            _, missing, unexpected, _, _ = load_checkpoint(
                **config.model.resume_from,
                model=model,
                model_ema=model_ema if not config.train.use_fsdp else None,
                FSDP=config.train.use_fsdp,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                null_embed_path=null_embed_path,
            )
            # one_logger_callbacks.on_load_checkpoint_end()
            logger.warning(f"Missing keys: {missing}")
            logger.warning(f"Unexpected keys: {unexpected}")

            path = osp.basename(config.model.resume_from["checkpoint"])
            try:
                start_epoch = int(path.replace(".pth", "").split("_")[1]) - 1
                start_step = int(path.replace(".pth", "").split("_")[3])
                # start_epoch = 2
            except:
                pass

        elif config.model.load_model_ckpt:
            logger.warning(
                f"No checkpoint to resume, load checkpoint with model.load_model_ckpt function from {config.model.load_model_ckpt}"
            )
            # one_logger_callbacks.on_load_checkpoint_start()
            model.load_model_ckpt(
                config.model.load_model_ckpt,
                init_patch_embedding=config.model.init_patch_embedding,
                enable_lora=use_lora,
            )
            # one_logger_callbacks.on_load_checkpoint_end()

    logger.info("Finish loading checkpoint")
    # 8. Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    # model, text_encoder = accelerator.prepare(model, text_encoder)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model = accelerator.prepare(model)
    if model_ema is not None and not config.train.use_fsdp:
        model_ema = accelerator.prepare(model_ema)
    if not config.train.offload_text_encoder:
        text_encoder = accelerator.prepare(text_encoder)
    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)
    if config.distill is not None:
        teacher_model = accelerator.prepare(teacher_model)

    # load everything except model when resume
    if (
        config.train.use_fsdp
        and config.model.resume_from is not None
        and config.model.resume_from["checkpoint"] is not None
        and config.model.resume_from["resume_optimizer"]
        and config.model.resume_from["resume_lr_scheduler"]
    ):
        logger.info("FSDP resume: Loading optimizer, scheduler, scaler, random_states...")
        accelerator.load_state(
            os.path.join(config.model.resume_from["checkpoint"], "model"),
            state_dict_key=["optimizer", "scheduler", "scaler", "random_states"],
        )

    set_random_seed((start_step + 1) // config.train.save_model_steps + int(os.environ["LOCAL_RANK"]))
    logger.info(f'Set seed: {(start_step + 1) // config.train.save_model_steps + int(os.environ["LOCAL_RANK"])}')

    # Start Training
    # one_logger_callbacks.on_train_start(
    #     train_iterations_start=start_step,
    #     train_samples_start=start_step * config.train.train_batch_size * get_world_size(),
    # )
    train(
        config=config,
        args=args,
        accelerator=accelerator,
        model=model,
        model_ema=model_ema,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        train_diffusion=train_diffusion,
        logger=logger,
    )
    # one_logger_callbacks.on_train_end()
    # one_logger_callbacks.on_app_end()


if __name__ == "__main__":
    # Print WanModelConfig field names and the file where it's defined
    # import inspect
    # from dataclasses import is_dataclass, fields as dataclass_fields

    # try:
    #     wan_model_config_file = inspect.getfile(WanModelConfig)
    # except Exception as e:
    #     wan_model_config_file = f"Could not get file: {e}"

    # if is_dataclass(WanModelConfig):
    #     wan_fields = [f.name for f in dataclass_fields(WanModelConfig)]
    # elif hasattr(WanModelConfig, "__annotations__"):
    #     wan_fields = list(WanModelConfig.__annotations__.keys())
    # else:
    #     # Fallback: list public attributes
    #     wan_fields = [k for k in dir(WanModelConfig) if not k.startswith("_")]

    # print(f"WanModelConfig fields: {wan_fields}")
    # print(f"WanModelConfig defined in: {wan_model_config_file}")

    main()
