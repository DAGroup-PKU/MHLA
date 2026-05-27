# -*- coding: utf-8 -*-

from datasets import load_from_disk
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          Trainer)
import os
from typing import Optional
import inspect



try:
    # Load environment variables from .env if present
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# import fla  # noqa
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import fla
from fla.models.gla.configuration_gla import GLAConfig
from flame.data import DataCollatorForLanguageModeling
from flame.logging import LogCallback, get_logger
from flame.parser import get_train_args

AutoConfig.register("gla", GLAConfig)



from transformers import TrainerCallback
import torch

class NaNMonitorCallback(TrainerCallback):
    def __init__(self, model):
        self.model = model
        self.register_nan_hooks(model)

    def register_nan_hooks(self, model):
        def forward_nan_hook(module, input, output):
            if output is torch.Tensor and (torch.isnan(output).any() or torch.isinf(output).any()):
                print(f"[NaN HOOK] Forward NaN detected in {module.__class__.__name__}")

        def backward_nan_hook(module, grad_input, grad_output):
            if any(g is not None and (torch.isnan(g).any() or torch.isinf(g).any()) for g in grad_input):
                print(f"[NaN HOOK] Backward NaN detected in {module.__class__.__name__}")

        for name, module in model.named_modules():
            # 只在常见层注册（避免 embedding 等层打印太多）
            if any(k in name.lower() for k in ["attn", "attention", "mlp", "ff", "norm", "linear"]):
                module.register_forward_hook(forward_nan_hook)
                module.register_full_backward_hook(backward_nan_hook)

    def on_step_end(self, args, state, control, **kwargs):
        # 在每个 step 结束时检测梯度
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"[NaN HOOK] Gradient anomaly in {name}")
        return control

logger = get_logger(__name__)


def main():
    args = get_train_args()
    logger.info(args)

    # PyTorch 2.6 默认将 torch.load 的 weights_only 设为 True。
    # 这里在未显式传入该参数时，将其默认设为 False（仅当该参数存在时），
    # 以保证能恢复包含任意对象（如 DeepSpeed 状态）的历史 checkpoint。
    try:
        import torch  # type: ignore

        _orig_torch_load = torch.load  # type: ignore
        try:
            _supports_weights_only = "weights_only" in inspect.signature(_orig_torch_load).parameters
        except Exception:
            _supports_weights_only = False

        def _patched_torch_load(*args, **kwargs):  # type: ignore
            if _supports_weights_only and "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return _orig_torch_load(*args, **kwargs)

        torch.load = _patched_torch_load  # type: ignore
    except Exception:
        pass

    # Optional: login to Weights & Biases using API key from environment (.env)
    # Set WANDB_API_KEY in your .env to enable automatic login.
    wandb_api_key: Optional[str] = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        try:
            import wandb  # type: ignore
            wandb.login(key=wandb_api_key)
            logger.info("Weights & Biases logged in via WANDB_API_KEY from environment.")
        except Exception as e:
            logger.warning(f"Failed to login to Weights & Biases: {e}")
    print(args.tokenizer)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        use_fast=args.use_fast_tokenizer,
        trust_remote_code=True,
        add_bos_token=True,
        add_eos_token=False,
        # force_download=True,
        # local_files_only=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Add pad token: {}".format(tokenizer.pad_token))
    if args.from_config:
        logger.info("All model params are randomly initialized for from-scratch training.")
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(args.model_name_or_path))
    else:
        logger.info(f"Loading pretrained checkpoint {args.model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.train()

    trainable_params, all_param = model.num_parameters(only_trainable=True), model.num_parameters()
    logger.info(f"% of trainable params: {trainable_params:d} / {all_param:d} = {trainable_params / all_param:.2%}")
    logger.info(f"{tokenizer}\n{model}\n{model.config}")

    logger.info(f"Loading the `{args.split}` split directly from the cache {args.cache_dir}...")
    dataset = load_from_disk(args.cache_dir)
    logger.info(f"{dataset}")
    logger.info(f"Shuffling the dataset with seed {args.seed}")
    dataset = dataset.shuffle(seed=args.seed)
    logger.info("Creating the data collator")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, varlen=args.varlen)
    logger.info(f"{data_collator}")

    if args.lr_scheduler_type == 'cosine_with_min_lr':
        args.lr_scheduler_kwargs = {'min_lr_rate': 0.1}
    if args.lr_scheduler_type == 'warmup_stable_decay':
        args.lr_scheduler_kwargs = {
            'num_stable_steps': args.max_steps * 0.9 - args.warmup_steps,
            'num_decay_steps': args.max_steps * 0.1
        }

    trainer = Trainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[LogCallback(), NaNMonitorCallback(model)],
        train_dataset=dataset
    )

    results = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(trainer.args.output_dir)

    trainer.log_metrics("train", results.metrics)
    trainer.save_metrics("train", results.metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
