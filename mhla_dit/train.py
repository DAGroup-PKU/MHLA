# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from accelerate import Accelerator
import wandb
from dotenv import load_dotenv

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import utils

from download import find_model


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class CustomDataset(Dataset):
    def __init__(self, features_dir, labels_dir):
        self.features_dir = features_dir
        self.labels_dir = labels_dir

        self.features_files = sorted(os.listdir(features_dir))
        self.labels_files = sorted(os.listdir(labels_dir))

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]

        features = np.load(os.path.join(self.features_dir, feature_file))
        labels = np.load(os.path.join(self.labels_dir, label_file))
        return torch.from_numpy(features), torch.from_numpy(labels)

#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    """
    Trains a new DiT model.
    """
    load_dotenv()
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup accelerator:
    accelerator = Accelerator()
    device = accelerator.device

    # These are defined early so they can be updated by checkpoint loading
    train_steps = 0
    start_epoch = 0

    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        if args.experiment == None:
            experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        else:
            experiment_dir = f"{args.results_dir}/{args.experiment}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)       
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        if args.wandb:
            wandb_run_id = None
            wandb_id_file = f"{experiment_dir}/wandb_run_id.txt"
            if os.path.exists(wandb_id_file):
                with open(wandb_id_file, "r") as f:
                    wandb_run_id = f.read().strip()
            
            wandb.init(project=args.wandb_project, 
                       entity=args.wandb_entity, 
                       config=args, 
                       name=experiment_dir.split("/")[-1], 
                       id=wandb_run_id, 
                       resume="allow")
            
            # If it's a new run, save the run ID
            if wandb.run.resumed is False:
                 with open(wandb_id_file, "w") as f:
                    f.write(wandb.run.id)


    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        attn_type=args.attn_type,
        block_kwargs=args.model_kwargs,
        piecewise_patchembed=True,
    )
    # Note that parameter initialization is done within the DiT constructor
    
    # Setup optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    
    # Create an EMA of the model for use after training
    ema = deepcopy(model)

    # Resume from checkpoint if specified
    if args.resume_ckpt:
        if accelerator.is_main_process:
            logger.info(f"Resuming training from checkpoint: {args.resume_ckpt}")
        checkpoint = torch.load(args.resume_ckpt, map_location="cpu",weights_only=False)
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        opt.load_state_dict(checkpoint["opt"])
        # Load starting epoch and training steps
        start_epoch = checkpoint["epoch"] + 1
        train_steps = checkpoint["train_steps"]
        if accelerator.is_main_process:
            logger.info(f"Resumed from epoch {start_epoch - 1} at step {train_steps}")
    # Load pre-trained weights for finetuning if not resuming
    elif args.ckpt:
        if accelerator.is_main_process:
            logger.info(f"Loading checkpoint for finetuning from {args.ckpt}")
        ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
        state_dict = find_model(ckpt_path)
        
        # Remap keys for architecture mismatch between standard DiT and MHLA DiT
        new_state_dict = {}
        for k, v in state_dict.items():
            if 'attn.qkv' in k:
                new_k = k.replace('attn.qkv', 'attn.to_qkv')
                new_state_dict[new_k] = v
            elif 'attn.proj' in k:
                new_k = k.replace('attn.proj', 'attn.to_out.0')
                new_state_dict[new_k] = v
            else:
                new_state_dict[k] = v

        # Filter out MHLA conv weights we want to train from scratch
        filtered_state_dict = {k: v for k, v in new_state_dict.items() if 'piece_attn.conv' not in k}
        
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
        if accelerator.is_main_process:
            logger.info("Loaded checkpoint with remapped keys.")
            logger.info(f"Missing keys: {missing_keys}")
            logger.info(f"Unexpected keys: {unexpected_keys}")
    

    model = model.to(device)
    ema = ema.to(device)
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    if accelerator.is_main_process:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup data:
    features_dir = f"{args.feature_path}/imagenet256_features"
    labels_dir = f"{args.feature_path}/imagenet256_labels"
    dataset = CustomDataset(features_dir, labels_dir)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // accelerator.num_processes),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images ({args.feature_path})")

    # Prepare models for training:
    model, opt, loader = accelerator.prepare(model, opt, loader)
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode



    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()
    
    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs...")
  
    for epoch in range(start_epoch, args.epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        
        # Set the epoch for the sampler to ensure proper shuffling
        if hasattr(loader.sampler, 'set_epoch'):
            loader.sampler.set_epoch(epoch)
        
        epoch_iterator = iter(loader)
        


        for x, y in epoch_iterator:
            x = x.to(device)
            y = y.to(device)

            x_no_flip, x_flip = x.chunk(2, dim=1)  # Split batch into non-flipped and flipped images
            if np.random.randint(0, 2) == 0:
                x = x_no_flip
            else:
                x = x_flip
            x = x.squeeze(dim=1)
            y = y.squeeze(dim=1)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            accelerator.backward(loss)


            if getattr(args, "max_grad_norm", 0) and args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            
            for module in model.modules():
                if hasattr(module, 'piece_attn') and hasattr(module.piece_attn, 'conv'):
                    module.piece_attn.conv.weight.data.clamp_(0.0, 1.0)

            update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item() / accelerator.num_processes

                avg_loss = torch.tensor(avg_loss, device=accelerator.device)
                avg_loss = accelerator.reduce(avg_loss, reduction="sum")

                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")

                    if args.wandb:
                        wandb.log({
                            "loss": avg_loss,
                            "step": train_steps,
                            "epoch": epoch,
                            "steps_per_sec": steps_per_sec,
                        })
                running_loss = 0
                log_steps = 0
                start_time = time()

        # Save DiT checkpoint:
        if (epoch % args.ckpt_every == 0 and epoch > 0) or (epoch == args.epochs - 1):
            if accelerator.is_main_process:
                checkpoint = {
                    "model": model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args,
                    "epoch": epoch,
                    "train_steps": train_steps
                }
                checkpoint_path = f"{checkpoint_dir}/epoch-{epoch:04d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")


    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    if accelerator.is_main_process:
        logger.info("Training complete. Creating signal file.")
        signal_file_path = f"{experiment_dir}/training_completed.signal"
        with open(signal_file_path, "w") as f:
            pass # Create empty signal file
        if args.wandb:
            wandb.finish()
        logger.info("Done!")


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", type=str, default="features")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=100, help="Save a checkpoint every N epochs.")
    parser.add_argument('--model-kwargs', nargs='*', default={}, action=utils.ParseKwargs)
    parser.add_argument("--experiment", type=str, default=None) 
    parser.add_argument("--wandb", action="store_true", help="Enable logging to wandb")
    parser.add_argument("--wandb-project", type=str, default="DiT-MHLA", help="Wandb project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Wandb entity")
    parser.add_argument("--resume-ckpt", type=str, default=None, help="Path to a checkpoint to resume training from.")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to a checkpoint to finetune from.")
    parser.add_argument("--max-grad-norm", type=float, default=0.0, help="Gradient clipping max norm (set 0 to disable).")
    args = parser.parse_args()
    main(args)
