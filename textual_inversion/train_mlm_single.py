import inspect


from datasets_pkgs.dataset_mlm import TextualInversionDataset
from configs import parse_args
import sys
sys.path.insert(0, './packages')
import argparse
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import PIL
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder

# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
# from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
# from diffusers.utils.import_utils import is_xformers_available

# ADDED
from data_utils import cycle, create_wbd
from torch import nn
from utils import render_caption
# ADDED
if is_wandb_available():
    import wandb


# ------------------------------------------------------------------------------


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.28.0.dev0")

logger = get_logger(__name__)



def log_validation(tokenizer, args, accelerator, target_emb,pipeline,step):
    
    # create pipeline (note: unet and vae are loaded again in float32)
    

    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    # dog
    if args.include_prior_concept:
        placeholder='{} {}'.format(args.placeholder_token1,args.prior_concept1)
    else:
        placeholder='{}'.format(args.placeholder_token1)

    if args.prompt_type=='pet':
        validation_prompts=[
            "a picture of {} swimming in a pool".format(placeholder),
            "a picture of {} with the Great Wall of China in the background".format(placeholder),
            "a picture of {} in times square".format(placeholder),
            "{} on a boat in the sea".format(placeholder),
            "{} in a purple wizard outfit".format(placeholder),
            "{} playing with a ball".format(placeholder),
            "{} wearing sunglasses".format(placeholder),
            ]
    # vase
    
    elif args.prompt_type in ['nonliving']:
        validation_prompts = [
            'a {0} in the jungle'.format(placeholder),
            'a {0} in the snow'.format(placeholder),
            'a {0} with a blue house in the background'.format(placeholder),
            'a {0} with the Eiffel Tower in the background'.format(placeholder),
            'a purple {0}'.format(placeholder),
            'a wet {0}'.format(placeholder),
            'a cube shaped {0}'.format(placeholder)
            ]
    elif args.prompt_type in ['building']:
        validation_prompts = [
            '{} in snowy ice.'.format(placeholder),
            '{} at a beach with a view of the seashore.'.format(placeholder),
            'Photo of the {} with the sun rising in the sky.'.format(placeholder),
            'cat sitting in front of {} in snowy ice.'.format(placeholder),
            '{} digital painting 3d render geometric style.'.format(placeholder),
            'painting of {} in the style of van gogh.'.format(placeholder),
            'Top view of the {}. '.format(placeholder)
            ]
    elif args.prompt_type in ['sunglasses']:
        validation_prompts=[
            'photo of a {}'.format(placeholder),
            'close shot of {} on the sandy beach with a view of the seashore'.format(placeholder),
            'A scientist wearing {} examines a test tube'.format(placeholder),
            'A dog wearing {} on the porch'.format(placeholder),
            'A giraffe wearing {}'.format(placeholder),
            '{} painted in the style of andy warhol'.format(placeholder),
            'digital painting of a turtle wearing {}'.format(placeholder),
            '{} digital 3d render'.format(placeholder),
        ]
    else:
        assert False
    # print(validation_prompts[0],'validation_prompts')
    # print('Start Inference')
    is_keyword_tokens_list1=[]
    for prompt in validation_prompts:
        is_keyword_tokens1=[False]
        text_words=prompt.split()
        for word_idx in range(len(text_words)):
            cap_word=text_words[word_idx]
            word_token_ids=tokenizer.encode(cap_word,add_special_tokens=False)
            num_tokens=len(word_token_ids)
            for tok_id in word_token_ids:
                if args.placeholder_token1 in cap_word:
                    is_keyword_tokens1.append(True)
                else:
                    is_keyword_tokens1.append(False)
        for _ in range(len(is_keyword_tokens1),tokenizer.model_max_length):
            is_keyword_tokens1.append(False)
        assert len(is_keyword_tokens1)==tokenizer.model_max_length
        is_keyword_tokens1=torch.BoolTensor(is_keyword_tokens1)
        # print(is_keyword_tokens1.shape,'is_keyword_tokens1.shape')
        is_keyword_tokens_list1.append(is_keyword_tokens1)
    is_keyword_tokens_list1=torch.stack(is_keyword_tokens_list1)
    logger.info(
        f"STEP {step} Running validation... \n Generating {len(validation_prompts)} images with prompt:"
        f" {validation_prompts}.",main_process_only=True
    )
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)
    with autocast_ctx:
        images = pipeline(validation_prompts, num_inference_steps=25, generator=generator,
                          silent=args.silent,
                          inj_embeddings1=target_emb,
                          is_keyword_tokens1=is_keyword_tokens_list1).images
    print('Generated')


    del pipeline
    torch.cuda.empty_cache()
    return images,validation_prompts





def main():
    args = parse_args()
    dict_args=vars(args)
    exp_dir=os.path.join(args.output_dir,args.run_name)    
    logging_dir = os.path.join(exp_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=exp_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )
    if args.report_to=='wandb' and accelerator.is_main_process:
        run=wandb.init(project=args.project_name)
        wandb.run.name = args.run_name
        wandb.run.save()
        wandb.config.update(args)
    


    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=True)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        viz_dir = os.path.join(exp_dir,'viz')
        os.makedirs(viz_dir, exist_ok=True)
        codepath=os.path.join(exp_dir,'src')
        if os.path.exists(codepath):
            assert False
        os.makedirs(codepath,exist_ok=True)
        os.system('cp *.py {}'.format(codepath))
        os.system('cp datasets_pkgs {} -R'.format(codepath))
        os.system('cp packages {} -R'.format(codepath))
        sample_dir=os.path.join(exp_dir,'samples')
        ckpt_dir=os.path.join(exp_dir,'checkpoints')
        os.makedirs(ckpt_dir,exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)
        # 1. command
        command_path=os.path.join(codepath,'command.txt')
        command_file=open(command_path,'w')
        command_file.write('cwd\t{}\n'.format(os.getcwd()))
        print(command_path,'command_path')
        idx=0
        while idx<len(sys.argv):
            item=sys.argv[idx]
            print(item,'item')
            command_file.write('{}\n'.format(item))
            idx+=1
        command_file.close()
       

    # Load tokenizer
    if args.tokenizer_name: #  NO
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )
    # Class Augmenter
    # class Augmenter(nn.Module):
    #     def __init__(self, embed_dim, width_fac=4):
    #         super(Augmenter, self).__init__()
    #         self.W_ff1 = nn.Linear(embed_dim, width_fac * embed_dim)
    #         self.W_ff2 = nn.Linear(embed_dim * width_fac, embed_dim)
    #         self.relu = nn.ReLU()
    #     def forward(self, X):
    #         # Simple Feedforward network that projects into a higher space (by width_fac) and back to embed_dim
    #         X = self.W_ff1(X)
    #         X = self.relu(X)
    #         return self.W_ff2(X)
    # augmenter=Augmenter(embed_dim=768*2)
    # # # # # # # # # # 
    # HERE
    # Add the placeholder token in tokenizer
    mask_tokens = [args.mask_tokens]
    placeholder_tokens = [args.placeholder_token1]
    tokenizer.add_tokens(mask_tokens)
    tokenizer.add_tokens(placeholder_tokens)
    mask_token_ids = tokenizer.convert_tokens_to_ids(mask_tokens)
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
    initializer_token_ids = tokenizer.encode(args.prior_concept1, add_special_tokens=False)
    initializer_token_id = initializer_token_ids[0]
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    print(token_embeds.shape,'token_embeds.shape')
    prior_embed=token_embeds[initializer_token_id].detach().clone().unsqueeze(0)
    # Initializer
    with torch.no_grad():
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = token_embeds[initializer_token_id].clone()
    
    # mask_embeds=token_embeds[mask_token_ids]
    if args.mask_embed_path is not None:
        mask_embeds=torch.load(args.mask_embed_path)[args.mask_tokens].to(accelerator.device)
        mask_embeds=F.normalize(mask_embeds,p=1,dim=-1)*args.avg_norm
    
    # HERE
    # # # # # # # # # # 
    from contextnet import ContextNet
    cls_net=ContextNet(768, len(token_embeds))


    # Freeze vae and unet
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    # Freeze all parameters except for the token embeddings in text encoder
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
    if args.gradient_checkpointing: #FAlse
        # Keep unet in train mode if we are using gradient checkpointing to save memory.
        # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
        unet.train()
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()
    unet.eval()
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
    params_to_optimize = [
        {"params": text_encoder.get_input_embeddings().parameters(), "lr": args.learning_rate},
        # {"params": augmenter.parameters(), "lr": args.learning_rate},
        # {"params": learned_embeds, "lr": args.learning_rate},
        # {"params": cls_net.parameters(), "lr": args.learning_rate},
    ]
    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        params_to_optimize,  # only optimize the embeddings
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    print(mask_token_ids,'mask_token_ids')
    train_dataset = TextualInversionDataset(
        include_prior_concept=args.include_prior_concept,
        data_root=args.train_data_dir1,
        tokenizer=tokenizer,
        size=args.resolution,
        placeholder_token=(" ".join(tokenizer.convert_ids_to_tokens(placeholder_token_ids))),
        repeats=args.repeats,
        center_crop=args.center_crop,
        flip_p=args.flip_p,
        exclude_suffix=args.exclude_suffix,
        prior_concept=args.prior_concept1,
        mask_token_ids=mask_token_ids[0],
        mlm_target=args.mlm_target,
        get_images=True,
        prompt_type=args.prompt_type,
    )
    train_dataset_mlm = TextualInversionDataset(
        include_prior_concept=args.include_prior_concept,
        data_root=args.train_data_dir1,
        tokenizer=tokenizer,
        size=args.resolution,
        placeholder_token=(" ".join(tokenizer.convert_ids_to_tokens(placeholder_token_ids))),
        repeats=args.repeats,
        center_crop=args.center_crop,
        flip_p=args.flip_p,
        exclude_suffix=args.exclude_suffix,
        prior_concept=args.prior_concept1,
        mask_token_ids=mask_token_ids[0],
        mlm_target=args.mlm_target,
        get_images=False,
        prompt_type=args.prompt_type,
    )
    
    def collate_fn(examples):

        if 'pixel_values' in examples[0]:
            # 1. pixel_values
            pixel_values = [example["pixel_values"] for example in examples]
            pixel_values = torch.stack(pixel_values)
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            masks = [example["masks"] for example in examples]
            masks = torch.stack(masks)
            masks = masks.to(memory_format=torch.contiguous_format).float()

            # 2. input ids
            input_ids = [example["input_ids"] for example in examples]
            input_ids=torch.stack(input_ids)
            # 2. input ids
            is_keyword_tokens = [example["is_keyword_tokens"] for example in examples] #N,77, list of booleans
            is_keyword_tokens = torch.stack(is_keyword_tokens)
        else:
            pixel_values=[]
            input_ids=[]
            is_keyword_tokens=[]
            masks=[]

       
        # 5. For MLM 
        input_ids_masked = [example["input_ids_masked"] for example in examples]
        input_ids_masked=torch.stack(input_ids_masked)
        input_ids_non_mask = [example["input_ids_non_mask"] for example in examples]
        input_ids_non_mask=torch.stack(input_ids_non_mask)
        masked_idxs = [example["masked_idxs"] for example in examples] #N,77, list of booleans
        masked_idxs = torch.stack(masked_idxs)
        mlm_labels = [example["mlm_labels"] for example in examples] #N,77, list of booleans
        mlm_labels = torch.stack(mlm_labels)
        non_special_idxs = [example["non_special_idxs"] for example in examples] #N,77, list of booleans
        non_special_idxs = torch.stack(non_special_idxs)
        is_keyword_tokens_mlm = [example["is_keyword_tokens_mlm"] for example in examples] #N,77, list of booleans
        is_keyword_tokens_mlm = torch.stack(is_keyword_tokens_mlm)
        # 5. For MLM 


        batch = {
            "pixel_values": pixel_values,
            "input_ids": input_ids, # for reconstruction
            "input_ids_masked": input_ids_masked, # for mlm
            "input_ids_non_mask": input_ids_non_mask, # for mlm
            "masked_idxs": masked_idxs,
            "mlm_labels": mlm_labels,
            "non_special_idxs": non_special_idxs,
            "is_keyword_tokens_mlm": is_keyword_tokens_mlm,
            "is_keyword_tokens": is_keyword_tokens,
            "masks": masks,
        }
        return batch
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
    )
    mlm_loader = torch.utils.data.DataLoader(
            train_dataset_mlm,
            batch_size=args.mlm_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
        )
    mlm_loader = cycle(mlm_loader)
    def load_mlm_batch(mlm_loader):
        mlm_data=next(mlm_loader)
        return mlm_data


    if args.validation_epochs is not None:
        warnings.warn(
            f"FutureWarning: You are doing logging with validation_epochs={args.validation_epochs}."
            " Deprecated validation_epochs in favor of `validation_steps`"
            f"Setting `args.validation_steps` to {args.validation_epochs * len(train_dataset)}",
            FutureWarning,
            stacklevel=2,
        )
        args.validation_steps = args.validation_epochs * len(train_dataset)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
    )

    text_encoder.train()
    # Prepare everything with our `accelerator`.
    
    text_encoder, optimizer, train_dataloader, lr_scheduler,cls_net,mlm_loader = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler,cls_net,mlm_loader
    )
    

    if args.cls_net_path is not None:
        for defined_key in cls_net.state_dict():
            print(defined_key,'defined_key-clsnet')
        saved_state_dict = torch.load(args.cls_net_path, map_location=torch.device('cpu'))
        print()
        new_state_dict={}
        for saved_key in saved_state_dict:
            new_key=saved_key
            print(saved_key,'saved_key-clsnet')
            if accelerator.num_processes>1:
                if not saved_key.startswith('module.'):
                    new_key='module.'+saved_key
            else:
                if saved_key.startswith('module.'):
                    new_key=saved_key.replace('module.','')
            new_state_dict[new_key]=saved_state_dict[saved_key]
        cls_net.load_state_dict(new_state_dict,strict=True)
    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    # if accelerator.is_main_process:
    #     accelerator.init_trackers("Textual Inversion NonObj Original", config=vars(args))
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****",main_process_only=True)
    logger.info(f"  Num examples = {len(train_dataset)}",main_process_only=True)
    logger.info(f"  Num Epochs = {args.num_train_epochs}",main_process_only=True)
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}",main_process_only=True)
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}",main_process_only=True)
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}",main_process_only=True)
    logger.info(f"  Total optimization steps = {args.max_train_steps}",main_process_only=True)
    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save
    if not(args.silent):
        progress_bar = tqdm(
            range(0, args.max_train_steps),
            initial=0,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
        )
    # keep original embeddings as reference
    # pipeline = DiffusionPipeline.from_pretrained(
    #     args.pretrained_model_name_or_path,
    #     text_encoder=accelerator.unwrap_model(text_encoder),
    #     tokenizer=tokenizer,
    #     unet=unet,
    #     vae=vae,
    #     safety_checker=None,
    #     revision=args.revision,
    #     variant=args.variant,
    #     torch_dtype=weight_dtype,
    # )
    accepts_keep_fp32_wrapper = "keep_fp32_wrapper" in set(
                            inspect.signature(
                                accelerator.unwrap_model
                            ).parameters.keys()
                        )
    extra_args = (
        {"keep_fp32_wrapper": True}
        if accepts_keep_fp32_wrapper
        else {}
    )
    print(accepts_keep_fp32_wrapper,'accepts_keep_fp32_wrapper')
    print(extra_args,'extra_args')
    pipeline = StableDiffusionPipeline(
            vae=accelerator.unwrap_model(vae, **extra_args),
            unet=accelerator.unwrap_model(unet, **extra_args),
            text_encoder=accelerator.unwrap_model(text_encoder, **extra_args),
            tokenizer=accelerator.unwrap_model(tokenizer, **extra_args),
            scheduler=accelerator.unwrap_model(noise_scheduler, **extra_args),
            feature_extractor=None,
            safety_checker=None,
            requires_safety_checker=False,
        )
    # pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=False)
    orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()
    import time
    prior_embed=prior_embed.to(accelerator.device)
    for epoch in range(first_epoch, args.num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(text_encoder):
                # 1. Load Batch
                pixel_values=batch["pixel_values"]# B,77 list of booleans (tensor)
                input_ids=batch["input_ids"]# B,77 list of booleans (tensor)
                masks=batch["masks"]# B,77 list of booleans (tensor)
                masks64=torch.nn.functional.interpolate(masks,(64,64))
                is_keyword_tokens=batch["is_keyword_tokens"]# B,77 list of booleans (tensor)
                # for MLM
                batch_mlm=load_mlm_batch(mlm_loader)
                is_keyword_tokens_mlm=batch_mlm["is_keyword_tokens_mlm"]
                masked_idxs=batch_mlm["masked_idxs"]
                mlm_labels=batch_mlm["mlm_labels"].to(accelerator.device)
                non_special_idxs=batch_mlm["non_special_idxs"]
                input_ids_masked=batch_mlm["input_ids_masked"].to(accelerator.device)
                input_ids_non_mask=batch_mlm["input_ids_non_mask"].to(accelerator.device)
                # input_ids_non_mask=batch_mlm["input_ids_non_mask"]
                # 1. Load Batch
                
                # 2. Reconstruction Loss
                latents = vae.encode(pixel_values.to(dtype=weight_dtype)).latent_dist.sample().detach()
                latents = latents * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                learned_embeds=accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
                if args.normalize_target1:
                    target_emb=F.normalize(learned_embeds,p=1,dim=-1)*args.normalize_target1
                else:
                    target_emb=learned_embeds
                encoder_hidden_states = text_encoder(input_ids,
                                                     is_keyword_tokens1=is_keyword_tokens,
                                                     inj_embeddings1=target_emb
                                                     )[0].to(dtype=weight_dtype)
                
                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                if args.masked_loss:
                    model_pred=(model_pred*masks64)
                    target=(target*masks64)
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                # loss=loss.mean()
                
                # 3. MLM Loss
                loss_mlm=None
                if args.lambda_mlm:
                    clip_text_embedding_masked = text_encoder(input_ids_masked,
                                                            mask_embedding=mask_embeds.unsqueeze(0),
                                                            mask_idxs=masked_idxs,
                                                            is_keyword_tokens1=is_keyword_tokens_mlm,
                                                            inj_embeddings1=target_emb,
                                                            )[0].to(accelerator.device, dtype=weight_dtype)
                    mlm_logits=cls_net(clip_text_embedding_masked)
                    masked_idxs_flat=masked_idxs.view(-1)
                    loss_mlm = F.cross_entropy(
                        mlm_logits.view(-1,len(orig_embeds_params)),
                        mlm_labels.view(-1),
                        ignore_index=-100,
                        reduction='none'
                    )
                    loss_mlm[masked_idxs_flat]*=args.mlm_weight
                    loss_mlm=loss_mlm.mean()
                    loss+=(loss_mlm*args.lambda_mlm)

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Let's make sure we don't update any embedding weights besides the newly added token
                index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
                assert isinstance(mask_token_ids,list)
                assert isinstance(placeholder_token_ids,list)
                if args.freeze_mask_embedding:
                    index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False
                else:
                    # no_update = False -> update them
                    index_no_updates[min(placeholder_token_ids+mask_token_ids) : max(placeholder_token_ids+mask_token_ids) + 1] = False
                with torch.no_grad():
                    accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                        index_no_updates
                    ] = orig_embeds_params[index_no_updates]

            # Checks if the accelerator has performed an optimization step behind the scenes
            # if accelerator.sync_gradients:
            images = []
            if accelerator.is_main_process:
                if global_step % args.checkpointing_steps == 0:
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(ckpt_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]
                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints",main_process_only=True
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}",main_process_only=True)
                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(ckpt_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)
                    
                    learned_embeds_dict = {args.placeholder_token1: learned_embeds.detach().cpu()}
                    weight_name = f"learned_embeds_s{global_step}.pt"
                    # weight_name_augmenter = f"augmenter_s{global_step}.pt"
                    save_path = os.path.join(ckpt_dir, weight_name)
                    # save_path_augmenter = os.path.join(ckpt_dir, weight_name_augmenter)
                    torch.save(learned_embeds_dict, save_path)
                    # torch.save(augmenter.state_dict(), save_path_augmenter)
                if ((global_step % args.validation_steps == 0)):
                    # visualize input
                    input_image=(pixel_values[0].permute(1,2,0).detach().cpu().numpy()+1)*127.5
                    input_mask=masks[0].permute(1,2,0).detach().cpu().numpy()
                    if args.masked_loss:
                        input_image=input_image*input_mask
                    input_image=input_image.astype(np.uint8)
                    input_image=Image.fromarray(input_image)
                    input_image.save(os.path.join(viz_dir,'input_image_s{:05d}.jpg'.format(global_step)))
                    if args.lambda_mlm:
                        # 1. MLM Result Logging
                        viz_idx=0
                        masked_idxs=masked_idxs.detach().cpu().numpy()[viz_idx:viz_idx+1]
                        non_special_idxs=non_special_idxs.detach().cpu()[viz_idx:viz_idx+1]
                        mlm_logits=mlm_logits.argmax(-1).detach().cpu().numpy()[viz_idx:viz_idx+1]#1,77
                        input_ids_non_mask=input_ids_non_mask[viz_idx:viz_idx+1]
                        input_ids_masked=input_ids_masked[viz_idx:viz_idx+1]

                        input_ids_non_mask=input_ids_non_mask[non_special_idxs]
                        input_ids_masked=input_ids_masked[non_special_idxs]
                        mlm_logits=mlm_logits[non_special_idxs]
                        masked_idxs=masked_idxs[non_special_idxs]

                        decoded=tokenizer.batch_decode(input_ids_non_mask)
                        decoded_masked=tokenizer.batch_decode(input_ids_masked)
                        decoded_logits=tokenizer.batch_decode(mlm_logits)
                        decoded_list=[]
                        decoded_masked_list=[]
                        decoded_logits_list=[]
                        for d1,d2,d3,m in zip(decoded,decoded_masked,decoded_logits,masked_idxs):
                            if m:
                                decoded_list.append('{:10}'.format('M[{}]'.format(d1)))
                                decoded_masked_list.append('{:10}'.format(d2))
                                # decoded_masked_list.append('{:12}'.format('M[{}]'.format(d2)))
                                decoded_logits_list.append('{:10}'.format('M[{}]'.format(d3)))
                            else:
                                decoded_list.append('{:10}'.format(d1))
                                decoded_masked_list.append('{:10}'.format(d2))
                                decoded_logits_list.append('{:10}'.format(d3))
                        decoded=' '.join(decoded_list)
                        decoded_masked=' '.join(decoded_masked_list)
                        decoded_logits=' '.join(decoded_logits_list)
                        dots='-'*100
                        print()
                        print()
                        print(dots)
                        print(dots)
                        print('Step\t\t|{}'.format(global_step))
                        print('Raw\t\t|{}'.format(decoded))
                        print('Masked\t\t|{}'.format(decoded_masked))
                        print('Preds\t\t|{}'.format(decoded_logits))
                        print(dots)
                        print(dots)
                        print()
                        # 1. MLM Result Logging
                    if not args.debug:
                        # 2. Image Logging
                        with torch.no_grad():
                            learned_embeds=accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
                            if args.normalize_target1:
                                target_emb=F.normalize(learned_embeds,p=1,dim=-1)*args.normalize_target1
                            else:
                                target_emb=learned_embeds
                            # (tokenizer, args, accelerator, target_emb,pipeline,step)
                            images,validation_prompts = log_validation(
                                tokenizer=tokenizer, 
                                args=args, 
                                accelerator=accelerator, 
                                target_emb=target_emb,
                                pipeline=pipeline,
                                step=global_step
                            )

                        # save images
                        # validation_files=os.listdir(args.train_data_dir)
                        validation_files=sorted(os.listdir(args.train_data_dir1))
                        validation_target=Image.open(os.path.join((args.train_data_dir1),validation_files[0])).resize((512,512)).convert('RGB')

                        # mod here
                        num_images=len(images)
                        num_cols=num_images
                        num_rows=num_images//num_cols
                        margin_bottom=150
                        margin_right=10
                        merged_viz = Image.new('RGB', ((512+margin_right)*(num_cols+1), (512+margin_bottom)*num_rows), (255, 255, 255))
                        for ridx in range(num_rows):
                            merged_viz.paste(validation_target,(0,ridx*(512+margin_bottom)))
                        for iidx,(image, val_prompt) in enumerate(zip(images[:],validation_prompts[:])):
                            row_idx=iidx//num_cols
                            col_idx=iidx-(num_cols*row_idx)
                            x0=(col_idx+1)*(512+margin_right)
                            y0=row_idx*(512+margin_bottom)+512
                            x1=x0+(512+margin_right)
                            y1=y0+margin_bottom
                            merged_viz=render_caption(merged_viz,val_prompt,[x0,y0+20,x1,y1])
                            merged_viz.paste(image.convert('RGB'),((col_idx+1)*(512+margin_right),row_idx*(512+margin_bottom)))
                        merged_viz.save(os.path.join(sample_dir, 'sample_{:05d}.jpg'.format(global_step)))
                        # mod here


                        if args.report_to=='wandb':
                            if (global_step) % args.save_steps == 0:   
                                wandb_image = wandb.Image(merged_viz, caption="img_{:06d}_result.jpg".format(global_step))
                                run.log({"examples": wandb_image})
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                with torch.no_grad():
                    # target_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
                    # target_embeds_log=target_embeds.detach()
                    # if args.normalize_target1:
                    #     norm_target=torch.norm(learned_embeds_scaled,p=1,dim=-1)
                    # else:
                    norm_target=torch.norm(target_emb,p=1,dim=-1)
                if loss_mlm is not None:
                    logs['loss_mlm']=loss_mlm.detach().item()#*args.lambda3
                logs['norm_target']=norm_target.item()
                if args.report_to=='wandb' and accelerator.is_main_process:
                    wandb.log(logs)
                if args.silent:
                    log_keys=list(logs.keys())
                    for lk in log_keys:
                        print('{}:{:.4f}'.format(lk,logs[lk]),end='\t')
                    print('{}:{:.4f}'.format(lk,logs[lk]))
                else:
                    progress_bar.set_postfix(**logs)
                    global_step += 1
                    progress_bar.update(1)
            if global_step >= args.max_train_steps:
                break
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        weight_name = "learned_embeds_final.pt" 
        # weight_name_augmenter= "learned_embeds_final.pt" 
        save_path = os.path.join(ckpt_dir, weight_name)
        # save_path_augmenter = os.path.join(ckpt_dir, weight_name_augmenter)
        # learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
        learned_embeds_dict = {args.placeholder_token1: learned_embeds.detach().cpu()}
        torch.save(learned_embeds_dict, save_path)
        # torch.save(augmenter.state_dict(), save_path_augmenter)

        

    accelerator.end_training()


if __name__ == "__main__":
    main()