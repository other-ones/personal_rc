import os
import inspect

import sys
sys.path.insert(0, './packages')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from datasets_pkgs.dataset_pplus_clean import PPlusDataset
from configs import parse_args
import argparse
import logging
import math
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
    StableDiffusionPipelinePPlus,
    UNet2DConditionModelPPlus,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
# from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
# from diffusers.utils.import_utils import is_xformers_available

# ADDED
from data_utils import cycle, create_wbd
from torch import nn
from utils import render_caption
torch.use_deterministic_algorithms(True)
# ADDED
if is_wandb_available():
    import wandb


# ------------------------------------------------------------------------------


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.28.0.dev0")

logger = get_logger(__name__)



def log_validation(tokenizer, args, accelerator, target_emb,pipeline,step,placeholder_tokens):
    
    # create pipeline (note: unet and vae are loaded again in float32)
    

    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    # dog
    

    if args.eval_prompt_type=='living':
        validation_prompts=[
        'a {0} in the jungle',
        'a {0} with a city in the background',
        'a {0} with a mountain in the background',
        'a {0} on top of a purple rug in a forest',
        'a {0} in a chef outfit',
        'a {0} in a police outfit',
        'a cube shaped {0}'
        ]
    elif args.eval_prompt_type =='nonliving':
        validation_prompts = [
            'a {0} in the jungle',
            'a {0} with a city in the background',
            'a {0} with a mountain in the background',
            'a {0} with the Eiffel Tower in the background',
            'a {0} floating on top of water',
            'a {0} floating in an ocean of milk',
            'a {0} on top of the sidewalk in a crowded street',
            'a cube shaped {0}'
            ]
    else:
        assert False, 'undefined eval prompt type'

        
    # print(validation_prompts[0],'validation_prompts')
    # print('Start Inference')
    validation_prompts_list=[]
    viz_promps_list=[]
    for prompt in validation_prompts:
        for pidx in range(len(placeholder_tokens)):
            if args.include_prior_concept:
                placeholder='{} {}'.format(placeholder_tokens[pidx],args.train_prior_concept1)
            else:
                placeholder='{}'.format(placeholder_tokens[pidx])
            val_prompt=prompt.format(placeholder)
            validation_prompts_list.append(val_prompt)

    num_val=len(validation_prompts)
    emb_dim=target_emb.shape[-1]
    logger.info(
        f"STEP {step} Running validation... \n Generating {len(validation_prompts)} images with prompt:"
        f" {validation_prompts}.",main_process_only=True
    )
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)
    with autocast_ctx:
        images = pipeline(validation_prompts_list, 
                        num_inference_steps=25, generator=generator,
                        silent=args.silent,
                        num_vectors1=args.num_vectors1
                        ).images
    print('Generated')


    del pipeline
    torch.cuda.empty_cache()
    return images,validation_prompts_list[::args.num_vectors1]





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
        if os.path.exists(codepath) and 'tmp' not in codepath:
            assert False
        os.makedirs(codepath,exist_ok=True)
        os.system('cp *.py {}'.format(codepath))
        os.system('cp ../datasets_pkgs {} -R'.format(codepath))
        os.system('cp packages {} -R'.format(codepath))
        caption_log_path=os.path.join(codepath,'log_captions.txt')
        caption_log_file=open(caption_log_path,'w')
        # copy clip
        os.makedirs(os.path.join(codepath,'clip_src'),exist_ok=True)
        target = os.readlink('clip_src/modeling_clip.py')
        shutil.copy2(target, '{}/clip_src/modeling_clip.py'.format(codepath))
        target = os.readlink('clip_src/modeling_outputs.py')
        shutil.copy2(target, '{}/clip_src/modeling_outputs.py'.format(codepath))
        # copy clip
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
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=None
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=None, variant=args.variant
    )
    unet = UNet2DConditionModelPPlus.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=None, variant=args.variant
    )
    
    # # # # # # # # # # 
    # HERE
    # Add the placeholder token in tokenizer
    print(len(tokenizer),'before add tokens')
    mask_tokens = [args.mask_tokens]
    tokenizer.add_tokens(mask_tokens)
    if args.num_vectors1>1:
        placeholder_tokens = []
        for vidx in range(args.num_vectors1):
            placeholder_tokens.append(args.placeholder_token1+'_{}'.format(vidx))
    else:
        placeholder_tokens = [args.placeholder_token1]
    tokenizer.add_tokens(placeholder_tokens)
    print(len(tokenizer),'after add tokens')
    mask_token_ids = tokenizer.convert_tokens_to_ids(mask_tokens)
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
    
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    # Initializer
    if args.initializer_token:
        initializer_token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
        initializer_token_id = initializer_token_ids[0]
        with torch.no_grad():
            for token_id in placeholder_token_ids:
                token_embeds[token_id] = token_embeds[initializer_token_id].clone()
    
    # mask_embeds=token_embeds[mask_token_ids]
    if args.mask_embed_path is not None:
        mask_embeds=torch.load(args.mask_embed_path)[args.mask_tokens].to(accelerator.device)
        # mask_embeds=F.normalize(mask_embeds,p=1,dim=-1)*args.avg_norm
    
    # HERE
    # # # # # # # # # # 
    # from contextnet import ContextNet
    # # pretrained with one placeholder, one mask_embeds
    # cls_output_dim=len(token_embeds)-args.num_vectors1+1
    # cls_net=ContextNet(768, len(token_embeds)-args.num_vectors1+1)
    if args.lambda_mlm:
        if 'contextnetv6' in args.cls_net_path:
            from contextnet_v3 import ContextNetV3 as ContextNet
        else:
            from contextnet import ContextNet
        if 'stable-diffusion-2-1' in args.pretrained_model_name_or_path:
            hidden_dim=1024
        elif 'stable-diffusion-v1-5' in args.pretrained_model_name_or_path:
            hidden_dim=768
        cls_output_dim=len(token_embeds)-(args.num_vectors1)
        cls_net=ContextNet(hidden_dim, cls_output_dim) # -1 for placeholder


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
    if args.exclude_cap_types is not None:
        exclude_cap_types=args.exclude_cap_types.split('-')
    else:
        exclude_cap_types=None
    if args.check_tag:
        check_tag=args.check_tag.split('-')
    else:
        check_tag=None
    train_dataset = PPlusDataset(
        get_images=True,
        data_root=args.train_data_dir1,
        tokenizer=tokenizer,
        include_prior_concept=args.include_prior_concept,
        size=args.resolution,
        placeholder_tokens=placeholder_tokens, #for pplus
        center_crop=args.center_crop,
        flip_p=args.flip_p,
        train_prior_concept1=args.train_prior_concept1,
        mask_token_ids=mask_token_ids,
        mlm_target=args.mlm_target,
        prompt_type=args.train_prompt_type,
        mask_prob=args.mask_prob,
        caption_root=args.caption_root,
        seed=args.seed,
        exclude_cap_types=exclude_cap_types,
        target_image=args.target_image,
        check_tag=check_tag,
    )
    train_dataset_mlm = PPlusDataset(
        get_images=False,
        data_root=args.train_data_dir1,
        tokenizer=tokenizer,
        include_prior_concept=args.include_prior_concept,
        size=args.resolution,
        placeholder_tokens=placeholder_tokens, #for pplus
        center_crop=args.center_crop,
        flip_p=args.flip_p,
        train_prior_concept1=args.train_prior_concept1,
        mask_token_ids=mask_token_ids,
        mlm_target=args.mlm_target,
        prompt_type=args.train_prompt_type,
        mask_prob=args.mask_prob,
        caption_root=args.caption_root,
        seed=args.seed,
        exclude_cap_types=exclude_cap_types,
        target_image=args.target_image,
        check_tag=check_tag,

    )
    
    def collate_fn(examples):
        if 'pixel_values' in examples[0]:
            # 1. pixel_values
            pixel_values = [example["pixel_values"] for example in examples]
            pixel_values = torch.stack(pixel_values)
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

            # 2. input ids
            input_ids_list = [example["input_ids_list"] for example in examples]
            input_ids_list=torch.stack(input_ids_list) #1,77
            raw_captions_ti = [example["raw_caption_ti"] for example in examples]
            # 2. input ids
            # 5. For MLM 
            input_ids_masked_list = []
            input_ids_non_mask_list = []
            masked_idxs_list = []
            mlm_labels_list = []
            non_special_idxs_list = []
            raw_captions_mlm = []
            # 5. For MLM 
        else:
            pixel_values=[]
            input_ids_list=[]
            raw_captions_ti=[]

       
            # 5. For MLM 
            raw_captions_mlm = [example["raw_caption_mlm"] for example in examples]
            input_ids_masked_list = [example["input_ids_masked_list"] for example in examples]
            input_ids_masked_list=torch.stack(input_ids_masked_list)
            input_ids_non_mask_list = [example["input_ids_non_mask_list"] for example in examples]
            input_ids_non_mask_list=torch.stack(input_ids_non_mask_list)
            masked_idxs_list = [example["masked_idxs_list"] for example in examples] #N,77, list of booleans
            masked_idxs_list = torch.stack(masked_idxs_list)
            mlm_labels_list = [example["mlm_labels_list"] for example in examples] #N,77, list of booleans
            mlm_labels_list = torch.stack(mlm_labels_list)
            non_special_idxs_list = [example["non_special_idxs_list"] for example in examples] #N,77, list of booleans
            non_special_idxs_list = torch.stack(non_special_idxs_list)
            # 5. For MLM 


        batch = {
            "raw_captions_mlm": raw_captions_mlm, # for reconstruction
            "raw_captions_ti": raw_captions_ti, # for reconstruction
            "pixel_values": pixel_values, # for reconstruction
            "input_ids_list": input_ids_list, # for reconstruction
            "input_ids_masked_list": input_ids_masked_list, # for mlm
            "input_ids_non_mask_list": input_ids_non_mask_list, # for mlm
            "masked_idxs_list": masked_idxs_list,# for mlm
            "mlm_labels_list": mlm_labels_list,# for mlm
            "non_special_idxs_list": non_special_idxs_list,# for mlm
        }
        return batch
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
    )
    if args.lambda_mlm:
        mlm_loader = torch.utils.data.DataLoader(
                train_dataset_mlm,
                batch_size=args.mlm_batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=args.dataloader_num_workers*2,
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
    if args.lambda_mlm:
        text_encoder, optimizer, train_dataloader, lr_scheduler,cls_net,mlm_loader = accelerator.prepare(
            text_encoder, optimizer, train_dataloader, lr_scheduler,cls_net,mlm_loader
        )
    else:
        text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    
    assert args.cls_net_path is not None,'args.cls_net_path is not None'
    if args.lambda_mlm and args.cls_net_path is not None:
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
    pipeline = StableDiffusionPipelinePPlus(
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
    if args.mlm_idxs:
        mlm_idxs=[int(i) for i in args.mlm_idxs.split(',')]
    else:
        mlm_idxs=None
    for epoch in range(first_epoch, args.num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(text_encoder):
                # 1. Load Batch
                pixel_values=batch["pixel_values"]# B,77 list of booleans (tensor)
                raw_captions_ti=batch["raw_captions_ti"]# B,77 list of booleans (tensor)
                input_ids_list=batch["input_ids_list"]# B,77 list of booleans (tensor)

                
                
                # 1. Load Batch
                
                # 2. Reconstruction Loss
                latents = vae.encode(pixel_values.to(dtype=weight_dtype)).latent_dist.sample().detach()
                latents = latents * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                learned_embeds=accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] #9,768
                # learned_embeds: 9,768
                if args.normalize_target1:
                    target_emb=F.normalize(learned_embeds,p=1,dim=-1)*args.normalize_target1
                else:
                    target_emb=learned_embeds
                bsz,num_vectors,num_tokens=input_ids_list.shape
                input_ids_list=input_ids_list.reshape(-1,num_tokens)#n,9,77 -> 9n,77
                # print(target_emb.shape,'target_emb.shape train') #9,768
                emb_dim=target_emb.shape[-1]
                encoder_hidden_states = text_encoder(input_ids_list,
                                                     )[0].to(dtype=weight_dtype)
                # Predict the noise residual
                encoder_hidden_states=encoder_hidden_states.reshape(bsz,num_vectors,num_tokens,-1)
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                # loss=loss.mean()
                
                # 3. MLM Loss
                loss_mlm=None
                if args.lambda_mlm:
                    # # for MLM
                    # load mlm batch
                    batch_mlm=load_mlm_batch(mlm_loader)
                    input_ids_masked_list=batch_mlm["input_ids_masked_list"].to(accelerator.device)
                    masked_idxs_list=batch_mlm["masked_idxs_list"]
                    mlm_labels_list=batch_mlm["mlm_labels_list"].to(accelerator.device)
                    non_special_idxs_list=batch_mlm["non_special_idxs_list"]
                    raw_captions_mlm=batch_mlm["raw_captions_mlm"] # B,77 list of booleans (tensor)
                    input_ids_non_mask_list=batch_mlm["input_ids_non_mask_list"].to(accelerator.device)
                    
                    mlm_bsz=len(masked_idxs_list)
                    if mlm_idxs:
                        input_ids_masked_list=input_ids_masked_list[:,mlm_idxs,:]
                        mlm_labels_list=mlm_labels_list[:,mlm_idxs]
                    clip_text_embedding_masked = text_encoder(input_ids_masked_list.reshape(-1,num_tokens),
                                                            )[0].to(accelerator.device, dtype=weight_dtype)
                    # torch.Size([10, 7, 77]) input_ids_masked_list.shape
                    # torch.Size([10, 7, 77]) mlm_labels_list.shape
                    # torch.Size([70, 77, 49409]) mlm_logits_list.shape


                    mlm_logits_list=cls_net(clip_text_embedding_masked)
                    # masked_idxs_flat_list=masked_idxs.view(-1)
                    mlm_labels_list_flat=mlm_labels_list.view(-1)
                    loss_mlm = F.cross_entropy(
                        mlm_logits_list.view(-1,cls_output_dim),
                        mlm_labels_list.view(-1),
                        # ignore_index=-100,
                        reduction='none'
                    )
                    loss_mlm=loss_mlm[mlm_labels_list_flat!=(-100)]
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
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                # [1] CHECKPOINTING
                if(global_step % args.checkpointing_steps==0) or (global_step==1):
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
                    learned_embeds_saved=accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
                    learned_embeds_saved=learned_embeds_saved.clone().detach().cpu()
                    learned_embeds_dict = {args.placeholder_token1: learned_embeds_saved}
                    weight_name = f"learned_embeds_s{global_step}.pt"
                    save_path = os.path.join(ckpt_dir, weight_name)
                    torch.save(learned_embeds_dict, save_path)
                    del learned_embeds_saved
                # [1] CHECKPOINTING

                if accelerator.is_main_process:
                    # [2] CAPTION LOGGING
                    if ((global_step % args.log_steps == 0) or global_step==1) and accelerator.is_main_process:
                        caption_log_file=open(caption_log_path,'a')
                        for raw_caption_ti in raw_captions_ti:
                            caption_log_file.write('STEP{:04d}\t{}\n'.format(global_step,raw_caption_ti))
                            caption_log_file.flush()
                        caption_log_file.write('\n')
                        caption_log_file.flush()
                        if args.lambda_mlm:
                            for raw_caption_mlm in raw_captions_mlm:
                                caption_log_file.write('STEP{:04d}\t{}\n'.format(global_step,raw_caption_mlm))
                                caption_log_file.flush()
                            caption_log_file.write('\n')
                        caption_log_file.write('\n')
                        caption_log_file.flush()
                        caption_log_file.close()
                    # [2] CAPTION LOGGING
                    if ((global_step % args.validation_steps == 0) or global_step==1):
                        # [3] INPUT LOGGING
                        input_image=(pixel_values[0].permute(1,2,0).detach().cpu().numpy()+1)*127.5
                        input_image=input_image.astype(np.uint8)
                        input_image=Image.fromarray(input_image)
                        input_image.save(os.path.join(viz_dir,'input_image_s{:05d}.jpg'.format(global_step)))
                        # [3] INPUT LOGGING

                        # [4] MLM LOGGING
                        if args.lambda_mlm:
                            viz_batch_idx=0
                            if mlm_idxs:
                                num_vectors_viz=len(mlm_idxs)
                                viz_vec_idx=mlm_idxs[0]
                            else:
                                num_vectors_viz=num_vectors
                                viz_vec_idx=0
                            mlm_logits_list=mlm_logits_list.reshape(mlm_bsz,num_vectors_viz,num_tokens,cls_output_dim)

                            # non_special_idxs_list:  mlm_bsz,num_vectors,77
                            # input_ids_non_mask_list:mlm_bsz,num_vectors,77
                            # mlm_logits_list:        mlm_bsz,num_vectors,len(mlm_idxs),77
                            # input_ids_masked_list:  mlm_bsz,num_vectors,len(mlm_idxs),77

                            # SELECTIVE
                            # torch.Size([10, 7, 77]) non_special_idxs_list.shape
                            # torch.Size([10, 7, 77]) input_ids_non_mask_list.shape
                            # torch.Size([10, 3, 77, 49409]) mlm_logits_list.shape
                            # torch.Size([10, 3, 77]) input_ids_masked_list.shape
                            # SELECTIVE

                            # NON SELECTIVE
                            # torch.Size([10, 7, 77]) non_special_idxs_list.shape
                            # torch.Size([10, 7, 77]) input_ids_non_mask_list.shape
                            # torch.Size([10, 7, 77, 49409]) mlm_logits_list.shape
                            # torch.Size([10, 7, 77]) input_ids_masked_list.shape
                            # torch.Size([10, 7, 77]) mlm_labels_list.shape
                            # NON SELECTIVE

                            masked_idxs=masked_idxs_list[viz_batch_idx:viz_batch_idx+1,viz_vec_idx] #50*9,77
                            non_special_idxs=non_special_idxs_list[viz_batch_idx:viz_batch_idx+1,viz_vec_idx]
                            input_ids_non_mask=input_ids_non_mask_list[viz_batch_idx:viz_batch_idx+1,viz_vec_idx]
                            mlm_logits=mlm_logits_list[viz_batch_idx:viz_batch_idx+1,0]
                            input_ids_masked=input_ids_masked_list[viz_batch_idx:viz_batch_idx+1,0]
                            mlm_labels=mlm_labels_list[viz_batch_idx:viz_batch_idx+1,0]

                            mlm_logits=mlm_logits.argmax(-1).detach().cpu().numpy()#1,77

                            input_ids_non_mask=input_ids_non_mask[non_special_idxs]
                            input_ids_masked=input_ids_masked[non_special_idxs]
                            mlm_logits=mlm_logits[non_special_idxs]
                            masked_idxs=masked_idxs[non_special_idxs]
                            mlm_labels=mlm_labels[non_special_idxs].detach().cpu().numpy()
                            mlm_labels=mlm_labels[mlm_labels>0]


                            decoded=tokenizer.batch_decode(input_ids_non_mask)
                            decoded_masked=tokenizer.batch_decode(input_ids_masked)
                            decoded_logits=tokenizer.batch_decode(mlm_logits)
                            decoded_labels=tokenizer.batch_decode(mlm_labels)
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
                            decoded_lables=' '.join(decoded_labels)

                            dots='-'*100
                            print()
                            print()
                            print(dots)
                            print('Step\t\t|{}'.format(global_step))
                            print(dots)
                            print('Raw\t\t|{}'.format(decoded))
                            print('Masked\t\t|{}'.format(decoded_masked))
                            print('Preds\t\t|{}'.format(decoded_logits))
                            print(dots)
                            print('Labels\t\t|{}'.format(decoded_lables))
                            print(dots)
                            print()

                        # [4] MLM LOGGING


                        # [5] VALIDTION
                        if not args.debug:
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
                                    target_emb=target_emb, #9,768 ->9,n,768->9n,768,
                                    pipeline=pipeline,
                                    step=global_step,
                                    placeholder_tokens=placeholder_tokens,
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
                        # [5] VALIDTION


                        if args.report_to=='wandb':
                            if (global_step) % args.save_steps == 0:   
                                wandb_image = wandb.Image(merged_viz, caption="img_{:06d}_result.jpg".format(global_step))
                                run.log({"examples": wandb_image})
                
            # sync_grad
            # [6] PBAR PRINTING
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            with torch.no_grad():
                target_embeds_list_log = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
                target_embeds_list_log=target_embeds_list_log.detach() #7,768
                norm_target_list=torch.norm(target_embeds_list_log,p=1,dim=-1)
                for nti in range(args.num_vectors1):
                    logs['norm_target{}'.format(nti)]=norm_target_list[nti].item()
                #     # logs['norm_target{}'.format(1)]=norm_target[0].item()
                #     # logs['norm_target{}'.format(2)]=norm_target[2].item()
                #     # logs['norm_target{}'.format(3)]=norm_target[3].item()
                #     # logs['norm_target{}'.format(4)]=norm_target[4].item()
                #     # logs['norm_target{}'.format(8)]=norm_target[8].item()
            if loss_mlm is not None:
                logs['loss_mlm']=loss_mlm.detach().item()#*args.lambda3
            
            if args.report_to=='wandb' and accelerator.is_main_process:
                wandb.log(logs)
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            # [6] PBAR PRINTING
            if global_step >= args.max_train_steps:
                break
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    # if accelerator.is_main_process:
    #     weight_name = "learned_embeds_final.pt" 
    #     # weight_name_augmenter= "learned_embeds_final.pt" 
    #     save_path = os.path.join(ckpt_dir, weight_name)
    #     # save_path_augmenter = os.path.join(ckpt_dir, weight_name_augmenter)
    #     # learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
    #     learned_embeds_dict = {args.placeholder_token1: learned_embeds.detach().cpu()}
    #     torch.save(learned_embeds_dict, save_path)
    #     # torch.save(augmenter.state_dict(), save_path_augmenter)

        

    accelerator.end_training()


if __name__ == "__main__":
    main()