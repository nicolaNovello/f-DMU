# This code is modified from the Huggingface repository: https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora.py, and
import argparse
import hashlib
import itertools
import json
import logging
import math
import os
import sys
from typing import Callable, Dict
import warnings
from pathlib import Path
import flosses
import numpy as np
import torch
print(torch.__version__)
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import HfApi, create_repo
from model_pipeline import (
    CustomDiffusionAttnProcessor,
    CustomDiffusionPipeline,
    set_use_memory_efficient_attention_xformers,
)
from packaging import version
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from utils import *
from evaluate import *

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
if version.parse(diffusers.__version__) < version.parse("0.20.0"):
    from diffusers.models.cross_attention import CrossAttention
else:
    from diffusers.models.attention import Attention as CrossAttention

from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.14.0")
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn


import torch
import torch.nn as nn

import clip



logger = get_logger(__name__)

def perform_gradient_surgery_with_accumulation(
    model,
    accelerator,
    loss_a,
    loss_b,
    lambda_=1.0,
):
    """
    Performs gradient surgery compatible with Hugging Face Accelerate's gradient accumulation.

    This function should be called within the `accelerator.accumulate()` context and
    replaces the standard `accelerator.backward(loss)` call.

    It computes gradients for two losses (loss_a and loss_b) for the current micro-batch,
    adjusts the gradients from loss_a to be orthogonal to conflicting gradients from loss_b,
    and correctly accumulates the result with previously accumulated gradients.
    """
    # Get the underlying model if it's wrapped by accelerator 
    unwrapped_model = accelerator.unwrap_model(model)

    # Store gradients accumulated from previous micro-batches
    accumulated_grads = {}
    for name, p in unwrapped_model.named_parameters():
        if p.grad is not None:
            accumulated_grads[name] = p.grad.clone()

    # Clear gradients to compute grad(loss_b) for the current micro-batch
    unwrapped_model.zero_grad(set_to_none=True)

    # Compute gradients for loss_b for the current micro-batch.
    # accelerator.backward handles loss scaling for mixed precision and accumulation.
    # The `accelerator.accumulate` context manager handles `no_sync` correctly for DDP.
    accelerator.backward(loss_b, retain_graph=True)

    # Store the fresh gradients for loss_b for the relevant layers
    b_grads = {}
    for name, p in unwrapped_model.named_parameters():
        if "attn2" in name and p.grad is not None:
            b_grads[name] = p.grad.clone()

    # Clear gradients again to compute grad(loss_a)
    unwrapped_model.zero_grad(set_to_none=True)

    # Compute gradients for loss_a for the current micro-batch
    accelerator.backward(loss_a)

    # Now, p.grad contains the fresh gradient for loss_a. Perform the surgery.
    # The adjustment is only applied to the "attn2" cross-attention layers.
    for name, p in unwrapped_model.named_parameters():
        if name in b_grads and p.grad is not None:
            a_grad = p.grad
            b_grad = b_grads[name]

            # Normalize gradients
            b_grad_norm = b_grad / (torch.linalg.norm(b_grad) + 1e-8)
            a_grad_norm = a_grad / (torch.linalg.norm(a_grad) + 1e-8)

            # Calculate dot product
            dot_product = torch.dot(a_grad_norm.flatten(), b_grad_norm.flatten())

            # If gradients conflict, adjust the current grad(loss_a)
            if dot_product < 0:
                adjustment = lambda_ * dot_product * b_grad_norm
                p.grad -= adjustment

    # Add back the gradients from previous accumulation steps to the adjusted gradient
    for name, p in unwrapped_model.named_parameters():
        if name in accumulated_grads and p.grad is not None:
            p.grad += accumulated_grads[name]

def create_custom_diffusion(unet, parameter_group):
    for name, params in unet.named_parameters():
        if parameter_group == "cross-attn":
            if "attn2.to_k" in name or "attn2.to_v" in name:
                params.requires_grad = True
            else:
                params.requires_grad = False
        elif parameter_group == "attn":
            if "to_q" in name or "to_k" in name or "to_v" in name or "to_out" in name:
                print(name)
                params.requires_grad = True
            else:
                params.requires_grad = False
        elif parameter_group == "full-weight":
            params.requires_grad = True
        elif parameter_group == "embedding":
            params.requires_grad = False
        else:
            raise ValueError(
                "parameter_group argument only cross-attn, full-weight, embedding"
            )

    def change_attn(unet):
        for layer in unet.children():
            if type(layer) == CrossAttention:
                bound_method = set_use_memory_efficient_attention_xformers.__get__(
                    layer, layer.__class__
                )
                setattr(
                    layer, "set_use_memory_efficient_attention_xformers", bound_method
                )
            else:
                change_attn(layer)

    change_attn(unet)
    unet.set_attn_processor(CustomDiffusionAttnProcessor())
    return unet


def save_model_card(
    repo_id: str, images=None, base_model=str, prompt=str, repo_folder=None
):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"./image_{i}.png\n"

    yaml = f"""
        ---
        license: creativeml-openrail-m
        base_model: {base_model}
        instance_prompt: {prompt}
        tags:
        - stable-diffusion
        - stable-diffusion-diffusers
        - text-to-image
        - diffusers
        - custom diffusion
        inference: true
        ---
            """
    model_card = f"""
        # Custom Diffusion - {repo_id}

        These are Custom Diffusion adaption weights for {base_model}. The weights were trained on {prompt} using [Custom Diffusion](https://www.cs.cmu.edu/~custom-diffusion). You can find some example images in the following. \n
        {img_str[0]}
        """
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "OpenCLIPTextModel":
        from transformers import OpenCLIPTextModel
        return OpenCLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import (
            RobertaSeriesModelWithTransformation,
        )

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def freeze_params(params):
    for param in params:
        param.requires_grad = False


import torch
import torch.nn as nn


class ConvDiscriminator(nn.Module):
    def __init__(self, input_channels: int, base_channels: int = 64, num_layers: int = 3, use_spectral_norm: bool = True):
        super().__init__()

        if use_spectral_norm:
            layer = lambda in_c, out_c, K, S, P: nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, kernel_size=K, stride=S, padding=P))
        else:
            layer = lambda in_c, out_c, K, S, P: nn.Conv2d(in_c, out_c, kernel_size=K, stride=S, padding=P)

        sequence = [
            layer(input_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        mult = 1
        for n in range(1, num_layers):
            mult_prev = mult
            mult = min(2 ** n, 8)
            sequence += [
                layer(base_channels * mult_prev, base_channels * mult, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        mult_prev = mult
        mult = min(2 ** num_layers, 8)
        
        sequence += [
            layer(base_channels * mult_prev, base_channels * mult, 4, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * mult, 1, kernel_size=4, stride=1, padding=1) 
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        mask = mask.float()
        
        combined_input = torch.cat([x, mask], dim=1)
        
        return self.model(combined_input)
    
def importance_sampling_fn(t, temperature=0.05):
    return 1 / (1 + np.exp(-temperature * (t - 200))) - 1 / (1 + np.exp(-temperature * (t - 400)))    

    
def main(args):
    CLIP_GRADIENTS = args.clip_gradients
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        print(vars(args))
        accelerator.init_trackers("concept-ablation", config=vars(args))

    if args.seed is not None:
        set_seed(args.seed)
    if args.concepts_list is None:
        args.concepts_list = [
            {
                "instance_prompt": args.instance_prompt, 
                "class_prompt": args.class_prompt, 
                "instance_data_dir": args.instance_data_dir, 
                "class_data_dir": args.class_data_dir, 
                "caption_target": args.caption_target, 
            }
        ]
    else:
        with open(args.concepts_list, "r") as f:
            args.concepts_list = json.load(f)

    for i, concept in enumerate(args.concepts_list):

        if (concept["instance_prompt"] is not None and concept["instance_data_dir"] is not None):
            break
        class_images_dir = Path(concept["class_data_dir"])
        if not class_images_dir.exists(): 
            class_images_dir.mkdir(parents=True, exist_ok=True)
        os.makedirs(f"{class_images_dir}/images", exist_ok=True)

        if (len(list(Path(os.path.join(class_images_dir, "images")).iterdir())) < args.num_class_images):
            torch_dtype = (torch.float16 if accelerator.device.type == "cuda" else torch.float32)
            if args.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif args.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif args.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16
            print("TRE")
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=args.revision,
            )
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

            pipeline.set_progress_bar_config(disable=True)
            pipeline.to(accelerator.device)
            if not os.path.isfile(concept["class_prompt"]):

                if args.concept_type in ["style"]:

                    with open('../assets/finetune_prompts/painting.txt', 'r') as f:
                            class_prompt_collection = [x.strip() for x in f.readlines()]
                elif args.concept_type in ["nudity", "inappropriate_content"]:

                    with open('../assets/finetune_prompts/people.txt', 'r') as f:
                            class_prompt_collection = [x.strip() for x in f.readlines()]
                elif args.class_prompt in ["robot", "cat", "fish", "dog"]:

                    with open(f'../assets/finetune_prompts/{args.class_prompt}.txt', 'r') as f:
                            class_prompt_collection = [x.strip() for x in f.readlines()]
                # LLM based prompt collection for ablating new objects or memorization images
                else:

                    class_prompt = concept["class_prompt"]
                    # in case of object query chatGPT to generate captions containing the anchor category
                    class_prompt_collection, caption_target = getanchorprompts(
                        pipeline,
                        accelerator,
                        class_prompt,
                        args.concept_type,
                        class_images_dir,
                        args.num_class_prompts,
                        mem_impath=args.mem_impath if args.concept_type == "memorization" else None,
                        model_id=args.prompt_gen_model
                    )
                    concept["caption_target"] += f";*+{caption_target}"
                    with open(class_images_dir / "caption_target.txt", "w") as f:
                        f.write(concept["caption_target"])
                    print(class_prompt_collection, concept["caption_target"])
            else: 
                with open(concept["class_prompt"]) as f:
                    class_prompt_collection = [x.strip() for x in f.readlines()]
           
            num_new_images = args.num_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(class_prompt_collection, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)
            sample_dataloader = accelerator.prepare(sample_dataloader)

            if os.path.exists(f"{class_images_dir}/caption.txt"):
                os.remove(f"{class_images_dir}/caption.txt")
            if os.path.exists(f"{class_images_dir}/images.txt"):
                os.remove(f"{class_images_dir}/images.txt")

            for example in tqdm(
                sample_dataloader,
                desc="Generating class images",
                disable=not accelerator.is_local_main_process,
            ):
                accelerator.wait_for_everyone()
                with open(f"{class_images_dir}/caption.txt", "a") as f1, open(f"{class_images_dir}/images.txt", "a") as f2:
                    images = pipeline(
                        example["prompt"],
                        num_inference_steps=25,
                        guidance_scale=6.0,
                        negative_prompt=[args.caption_target]*len(example["prompt"]) if args.concept_type in ["nudity", "inappropriate_content"] else None,
                        eta=1.0,
                    ).images  # with this command it is generating the images

                    for i, image in enumerate(images):
                        hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                        image_filename = (
                            class_images_dir
                            / f"images/{example['index'][i]}-{hash_image}.jpg"
                        )
                        image.save(image_filename)
                        f2.write(str(image_filename) + "\n")
                    f1.write("\n".join(example["prompt"]) + "\n")
                    accelerator.wait_for_everyone()
            del pipeline
        print("QUO?")
        if args.concept_type == "memorization":
            filter(
                class_images_dir,
                args.mem_impath,
                outpath=str(class_images_dir / "filtered"),
            )
            with open(class_images_dir / "caption_target.txt", "r") as f:
                concept["caption_target"] = f.readlines()[0].strip()
            class_images_dir = class_images_dir / "filtered"

        concept["class_prompt"] = os.path.join(class_images_dir, "caption.txt")
        concept["class_data_dir"] = os.path.join(class_images_dir, "images.txt")
        concept["instance_prompt"] = os.path.join(class_images_dir, "caption.txt")
        concept["instance_data_dir"] = os.path.join(class_images_dir, "images.txt")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            print(args.hub_model_id or Path(args.output_dir).name)
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            )
            print(repo_id)
            repo_id = args.hub_model_id

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
            use_fast=False,
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)


    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet",
                                                revision=args.revision)
    vae.requires_grad_(False)
    if args.parameter_group != "embedding":
        text_encoder.requires_grad_(False)
    unet = create_custom_diffusion(unet, args.parameter_group)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16


    if accelerator.mixed_precision != "fp16":
        unet.to(accelerator.device, dtype=weight_dtype)
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.parameter_group == "embedding":
            text_encoder.gradient_checkpointing_enable()
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )
        if args.with_prior_preservation:
            args.learning_rate = args.learning_rate * 2.0

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Code taken from https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py
    modifier_token_id = []
    if args.parameter_group == "embedding":
        assert (
            args.concept_type != "memorization"
        ), "embedding finetuning is not supported for memorization"

        for concept in args.concept_list:
            token_ids = tokenizer.encode(
                [concept["caption_target"]], add_special_tokens=False
            )
        modifier_token_id += token_ids

        params_to_freeze = itertools.chain(
            text_encoder.text_model.encoder.parameters(),
            text_encoder.text_model.final_layer_norm.parameters(),
            text_encoder.text_model.embeddings.position_embedding.parameters(),
        )
        freeze_params(params_to_freeze)
        params_to_optimize = itertools.chain(
            text_encoder.get_input_embeddings().parameters()
        )
    else:
        if args.parameter_group == "cross-attn":
            params_to_optimize = itertools.chain(
                [
                    x[1]
                    for x in unet.named_parameters()
                    if ("attn2.to_k" in x[0] or "attn2.to_v" in x[0])
                ]
            )
        elif args.parameter_group == "attn":
            params_to_optimize = itertools.chain(
                [
                    x[1]
                    for x in unet.named_parameters()
                    if ("to_q" in x[0] or "to_k" in x[0] or "to_v" in x[0] or "to_out" in x[0] )
                ]
            )
        if args.parameter_group == "full-weight":
            params_to_optimize = itertools.chain(unet.parameters())
    # Optimizer creation
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = CustomDiffusionDataset(
        concepts_list=args.concepts_list,
        concept_type=args.concept_type,
        tokenizer=tokenizer,
        with_prior_preservation=args.with_prior_preservation,
        size=args.resolution,
        center_crop=args.center_crop,
        num_class_images=args.num_class_images,
        hflip=args.hflip,
        aug=not args.noaug,
        anchor_type=args.anchor_type,
        class_prompt = args.class_prompt
    )
    from functools import partial
    collate_with_args = partial(collate_fn, with_prior_preservation=args.with_prior_preservation)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_with_args,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    if args.variational:
        steps_warmup = 30
        args.max_train_steps += steps_warmup
        EMBEDDING_CHANNELS = 4 
        discriminator = ConvDiscriminator(input_channels=EMBEDDING_CHANNELS+1, base_channels = 64, num_layers = 3,
                                            use_spectral_norm=True)
        f_divergence_loss = flosses.FDivergenceLoss(args.f_divergence_type) # es. 'KL'
        optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr=0.0001)
        discriminator, optimizer_D = accelerator.prepare(discriminator, optimizer_D)
    else:
        steps_warmup=0
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
                                num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
                                num_training_steps=args.max_train_steps * args.gradient_accumulation_steps)


    if args.parameter_group == "embedding":
        text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    total_batch_size = (args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )

    if args.importance_sampling:
        print("""Using relation-focal importance sampling, which can make training more efficient
              and is particularly beneficial in erasing mass concepts with overlapping terms.""")

        list_of_candidates = [
            x for x in range(noise_scheduler.config.num_train_timesteps)
        ]
        prob_dist = [
            importance_sampling_fn(x)
            for x in list_of_candidates
        ]
        prob_sum = 0
        for j in prob_dist:
            prob_sum += j
        prob_dist = [x / prob_sum for x in prob_dist]
        
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    import pickle
    if not args.use_current_model_for_anchor:
        original_unet = pickle.loads(pickle.dumps(unet))
    gradient_tracker = GradientTracker()
    for epoch in range(first_epoch, args.num_train_epochs):
        if args.parameter_group == "embedding":
            text_encoder.train()
        else:
            unet.train()
        for step, batch in enumerate(train_dataloader):
            if (
                args.resume_from_checkpoint
                and epoch == first_epoch
                and step < resume_step
            ):
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(
                unet
            ) if args.parameter_group != "embedding" else accelerator.accumulate(
                text_encoder
            ):
                if args.parameter_group == "embedding":
                        text_encoder.train()
                else:
                        unet.train()
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=weight_dtype)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                if args.importance_sampling:
                        timesteps = np.random.choice(
                            list_of_candidates,
                            size=bsz,
                            replace=True,
                            p=prob_dist)
                        timesteps = torch.tensor(timesteps).cuda()
                else:
                        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
              
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(batch["input_ids"].cuda())[0]
                encoder_anchor_hidden_states = text_encoder(batch["input_anchor_ids"].cuda())[0]
                mask = batch["mask"]

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states.cuda()).sample
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    with torch.no_grad():
                        if not args.use_current_model_for_anchor:
                            model_pred_anchor = original_unet(
                                noisy_latents[: encoder_anchor_hidden_states.size(0)],
                                timesteps[: encoder_anchor_hidden_states.size(0)],
                                encoder_anchor_hidden_states.cuda().detach(),
                            ).sample
                        else:
                            model_pred_anchor = unet(
                            noisy_latents[: encoder_anchor_hidden_states.size(0)],
                            timesteps[: encoder_anchor_hidden_states.size(0)],
                            encoder_anchor_hidden_states.cuda().detach(),
                        ).sample


                if args.loss_type_reverse == "model-based":
                    if args.with_prior_preservation:
                        target_prior = torch.chunk(noise, 2, dim=0)[1]
                    target = model_pred_anchor
                else:
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(
                            f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                        )
                    if args.with_prior_preservation:
                        target, target_prior = torch.chunk(target, 2, dim=0)
                if args.with_prior_preservation:
                        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                        mask = torch.chunk(batch["mask"], 2, dim=0)[0]
                if args.variational:        
                    discriminator.train()


                    d_p_outputs = discriminator(target.detach(),mask.detach())
                    d_q_outputs = discriminator(model_pred.detach(),mask.detach())

                    loss_d = f_divergence_loss(d_p_outputs, d_q_outputs,mask)
                    

                    accelerator.backward(loss_d)

                    if accelerator.sync_gradients:
                        if CLIP_GRADIENTS:

                            accelerator.clip_grad_norm_(discriminator.parameters(), args.max_grad_norm)
                        optimizer_D.step()
                        optimizer_D.zero_grad()
                    discriminator.eval() 
                    if step >= steps_warmup or epoch >0:    
                        if args.with_prior_preservation:
                            d_q_outputs_g = discriminator(model_pred, mask)
                            loss_g =  f_divergence_loss.generator_loss(d_q_outputs_g)

                            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
                            perform_gradient_surgery_with_accumulation(
                                unet,
                                accelerator,
                                loss_g,
                                prior_loss,
                                lambda_=args.prior_loss_weight,
                            )
                            loss = loss_g + args.prior_loss_weight * prior_loss
                        else:
                            d_q_outputs_g = discriminator(model_pred, mask)
                            loss = f_divergence_loss.generator_loss( d_q_outputs_g)
                            accelerator.backward(loss)
                else:
                    if args.with_prior_preservation:
                        if args.f_divergence_type == "mse":
                            loss = F.mse_loss(
                                model_pred.float(), target.float(), reduction="none"
                            )
                            loss = ((loss * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])).mean()
                        elif args.f_divergence_type=="hellinger":
                            loss = flosses.MaskedSquaredHellingerLoss()( model_pred, target, mask)
                        elif args.f_divergence_type== "pearson_chi2":
                            loss = flosses.MaskedChiSquaredLoss()( model_pred, target, mask)
                        else:
                            assert "Error loss {} not supported without variational"

                        prior_loss = F.mse_loss(
                            model_pred_prior.float(), target_prior.float(), reduction="mean"
                        )

                        perform_gradient_surgery_with_accumulation(
                        unet,
                        accelerator,
                        loss,
                        prior_loss,
                        lambda_=args.prior_loss_weight,
                    )
                        loss = loss + args.prior_loss_weight * prior_loss
                    else:
                        mask = batch["mask"] 
                        if args.f_divergence_type == "mse_old":
                            loss = F.mse_loss(
                                model_pred.float(), target.float(), reduction="none"
                            )
                            loss = ((loss * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])).mean()
                        elif args.f_divergence_type == "mse":
                            loss = flosses.MaskedMSE()(model_pred, target, mask)
                        elif args.f_divergence_type=="hellinger":
                            loss = flosses.MaskedSquaredHellingerLoss()( model_pred, target, mask)
                        elif args.f_divergence_type== "pearson_chi2":
                            loss = flosses.MaskedChiSquaredLoss()( model_pred, target, mask)
                        else:
                            assert "Error loss {} not supported without variational"
                        accelerator.backward(loss)
                if not args.variational or step >= steps_warmup or epoch >0:
                    if args.parameter_group == "embedding":
                        if accelerator.num_processes > 1:
                            grads_text_encoder = (
                                text_encoder.module.get_input_embeddings().weight.grad
                            )
                        else:
                            grads_text_encoder = (
                                text_encoder.get_input_embeddings().weight.grad
                            )

                        index_grads_to_zero = (
                            torch.arange(len(tokenizer)) != modifier_token_id[0]
                        )
                        for i in range(len(modifier_token_id[1:])):
                            index_grads_to_zero = index_grads_to_zero & (
                                torch.arange(len(tokenizer)) != modifier_token_id[i]
                            )
                        grads_text_encoder.data[
                            index_grads_to_zero, :
                        ] = grads_text_encoder.data[index_grads_to_zero, :].fill_(0)

                    if accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(text_encoder.parameters())
                            if args.parameter_group == "embedding"
                            else itertools.chain(
                                [x[1] for x in unet.named_parameters() if ("attn2" in x[0])]
                            )
                            if args.parameter_group == "cross-attn"
                            else itertools.chain(unet.parameters())
                        )
                        
                        model_to_track = text_encoder if args.parameter_group == "embedding" else unet
                        mean_grad, std_grad = gradient_tracker.track_gradients(params_to_clip, epoch) 
                        if CLIP_GRADIENTS:
                            accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
            

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:

                    if accelerator.is_main_process:
                        pipeline = CustomDiffusionPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet=accelerator.unwrap_model(unet),
                            text_encoder=accelerator.unwrap_model(text_encoder),
                            tokenizer=tokenizer,
                            revision=args.revision,
                            modifier_token_id=modifier_token_id,
                        )
                        save_path = os.path.join(
                            args.output_dir, f"delta-{global_step}"
                        )
                        pipeline.save_pretrained(
                            save_path, parameter_group=args.parameter_group
                        )
                        logger.info(f"Saved state to {save_path}")
            if not args.variational:
                loss_d= torch.nan
            else:
                loss_d=loss_d.detach().item()
                if step< steps_warmup and epoch == 0:
                    loss= torch.nan
                else:
                    loss=loss.detach().item()
            logs = {"loss": loss,"loss_d": loss_d, "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if (
                args.validation_prompt is not None
                and global_step % args.validation_steps == 0
            ):
                logger.info(
                    f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                    f" {args.validation_prompt}."
                )
                # create pipeline
                pipeline = CustomDiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=accelerator.unwrap_model(unet),
                    text_encoder=accelerator.unwrap_model(text_encoder),
                    tokenizer=tokenizer,
                    revision=args.revision,
                    modifier_token_id=modifier_token_id,
                )
                pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipeline.scheduler.config
                )
                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)

                # run inference
                generator = torch.Generator(device=accelerator.device).manual_seed(
                    args.seed
                )
                images = [
                    pipeline(
                        args.validation_prompt,
                        num_inference_steps=25,
                        generator=generator,
                        eta=1.0,
                    ).images[0]
                    for _ in range(args.num_validation_images)
                ]

                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        np_images = np.stack([np.asarray(img) for img in images])
                        tracker.writer.add_images(
                            "validation", np_images, epoch, dataformats="NHWC"
                        )
                    if tracker.name == "wandb":
                        tracker.log(
                            {
                                "validation": [
                                    wandb.Image(
                                        image, caption=f"{i}: {args.validation_prompt}"
                                    )
                                    for i, image in enumerate(images)
                                ]
                            }
                        )

                del pipeline
                torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        pipeline = CustomDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            revision=args.revision,
            modifier_token_id=modifier_token_id,
        )
        save_path = os.path.join(args.output_dir, "delta.bin")
        pipeline.save_pretrained(save_path, parameter_group=args.parameter_group)

        # run inference
        if args.validation_prompt and args.num_validation_images > 0:
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                pipeline.scheduler.config
            )
            pipeline = pipeline.to(accelerator.device)
            pipeline.set_progress_bar_config(disable=True)

            # run inference
            generator = torch.Generator(device=accelerator.device).manual_seed(
                args.seed
            )
            images = [
                pipeline(
                    args.validation_prompt,
                    num_inference_steps=25,
                    generator=generator,
                    eta=1.0,
                ).images[0]
                for _ in range(args.num_validation_images)
            ]

            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images(
                        "test", np_images, epoch, dataformats="NHWC"
                    )
                if tracker.name == "wandb":
                    tracker.log(
                        {
                            "test": [
                                wandb.Image(
                                    image, caption=f"{i}: {args.validation_prompt}"
                                )
                                for i, image in enumerate(images)
                            ]
                        }
                    )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                prompt=args.instance_prompt,
                repo_folder=args.output_dir,
            )
            api = HfApi(token=args.hub_token)
            api.upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                path_in_repo=".",
                repo_type="model",
            )

    accelerator.end_training()


if __name__ == "__main__":
    import utils 
    args = utils.parse_args()
    main(args)


