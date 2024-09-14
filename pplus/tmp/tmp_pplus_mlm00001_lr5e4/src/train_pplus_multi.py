from datasets_pkgs.dataset_mlm_multi_contrastive import TextualInversionDatasetMulti
# from datasets_pkgs.dataset_mlm_multi_contrastive_backup import TextualInversionDatasetMulti

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
from utils import render_caption
import torch.nn.functional as F
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
# ADDED
if is_wandb_available():
    import wandb



# ------------------------------------------------------------------------------


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.28.0.dev0")

logger = get_logger(__name__)



def log_validation(tokenizer, args, accelerator, target_emb1,target_emb2,pipeline):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    # create pipeline (note: unet and vae are loaded again in float32)
    

    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    # dog
    if args.include_prior_concept:
        placeholder1='{} {}'.format(args.placeholder_token1,args.prior_concept1)
        placeholder2='{} {}'.format(args.placeholder_token2,args.prior_concept2)
    else:
        placeholder1='{}'.format(args.placeholder_token1)
        placeholder2='{}'.format(args.placeholder_token2)

    if args.prompt_type=='two_pets':
        validation_prompts=[
            "a picture of {} swimming in a pool".format(placeholder1),
            "a picture of {} swimming in a pool".format(placeholder2),
            "a picture of {} and {}".format(placeholder1,placeholder2),
            "a picture of {} and {} swimming in a pool".format(placeholder1,placeholder2),
            "a picture of {} and {} with the Great Wall of China in the background".format(placeholder1,placeholder2),
            "a picture of {} lying next to {}".format(placeholder1,placeholder2),
            "a picture of {} playing with {}".format(placeholder1,placeholder2),
            "a picture of {} chasing {}".format(placeholder1,placeholder2),
            # "a picture of {} and {} in times square".format(placeholder1,placeholder2),
            # "{} and {} on a boat in the sea".format(placeholder1,placeholder2),
            # "{} and {} in a purple wizard outfit".format(placeholder1,placeholder2),
            # "{} and {} playing with a ball".format(placeholder1,placeholder2),
            # "{} and {} wearing sunglasses".format(placeholder1,placeholder2),
            ]
    # vase
    else:
        assert False
    print(validation_prompts[0],'validation_prompts')
    print('Start Inference')
    is_keyword_tokens_list1=[]
    is_keyword_tokens_list2=[]
    for prompt in validation_prompts:
        is_keyword_tokens1=[False]
        is_keyword_tokens2=[False]
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
                if args.placeholder_token2 in cap_word:
                    is_keyword_tokens2.append(True)
                else:
                    is_keyword_tokens2.append(False)
        for _ in range(len(is_keyword_tokens1),tokenizer.model_max_length):
            is_keyword_tokens1.append(False)
        for _ in range(len(is_keyword_tokens2),tokenizer.model_max_length):
            is_keyword_tokens2.append(False)
        assert len(is_keyword_tokens1)==tokenizer.model_max_length
        assert len(is_keyword_tokens2)==tokenizer.model_max_length
        is_keyword_tokens1=torch.BoolTensor(is_keyword_tokens1)
        is_keyword_tokens2=torch.BoolTensor(is_keyword_tokens2)
        is_keyword_tokens_list1.append(is_keyword_tokens1)
        is_keyword_tokens_list2.append(is_keyword_tokens2)
    is_keyword_tokens_list1=torch.stack(is_keyword_tokens_list1)
    is_keyword_tokens_list2=torch.stack(is_keyword_tokens_list2)
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)
    with autocast_ctx:
        images = pipeline(validation_prompts, num_inference_steps=25, generator=generator,
                          inj_embeddings1=target_emb1,
                          inj_embeddings2=target_emb2,
                          is_keyword_tokens1=is_keyword_tokens_list1,
                          is_keyword_tokens2=is_keyword_tokens_list2
                          ).images
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
    logger.info(accelerator.state, main_process_only=False)
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
    
    # Add the placeholder token in tokenizer
    mask_tokens = [args.mask_tokens]
    placeholder_token1 = [args.placeholder_token1]
    placeholder_token2 = [args.placeholder_token2]
    print(placeholder_token1,'placeholder_token1')
    print(placeholder_token2,'placeholder_token2')
    tokenizer.add_tokens(mask_tokens)
    tokenizer.add_tokens(placeholder_token1)
    tokenizer.add_tokens(placeholder_token2)
    mask_token_ids = tokenizer.convert_tokens_to_ids(mask_tokens)
    placeholder_token_id1 = tokenizer.convert_tokens_to_ids(placeholder_token1)
    placeholder_token_id2 = tokenizer.convert_tokens_to_ids(placeholder_token2)
    initializer_token_id1 = tokenizer.encode(args.prior_concept1, add_special_tokens=False)
    initializer_token_id1 = initializer_token_id1[0]
    initializer_token_id2 = tokenizer.encode(args.prior_concept2, add_special_tokens=False)
    initializer_token_id2 = initializer_token_id2[0]
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for token_id in placeholder_token_id1:
            token_embeds[token_id] = token_embeds[initializer_token_id1].clone()
        for token_id in placeholder_token_id2:
            token_embeds[token_id] = token_embeds[initializer_token_id2].clone()
    print(token_embeds.shape,'token_embeds.shape')
    
    
    # mask_embeds=token_embeds[mask_token_ids]
    if args.mask_embed_path is not None:
        mask_embeds=torch.load(args.mask_embed_path)[args.mask_tokens].to(accelerator.device)
        mask_embeds=F.normalize(mask_embeds,p=1,dim=-1)*args.avg_norm
    # Add the placeholder token in tokenizer


    # Add learned concept
    if args.learned_embed_path1 and args.learned_embed_path2:
        learned_embed1=torch.load(args.learned_embed_path1)#[args.placeholder_token]
        learned_embed1=learned_embed1[args.placeholder_token1]
        learned_embed2=torch.load(args.learned_embed_path2)#[args.placeholder_token]
        learned_embed2=learned_embed2[args.placeholder_token2]
        # print(learned_embed1.shape,'learned_embed1.shape')
        # print(learned_embed2.shape,'learned_embed2.shape')
        # exit()
        with torch.no_grad():
            token_embeds[placeholder_token_id1] = learned_embed1.clone()
            token_embeds[placeholder_token_id2] = learned_embed2.clone()
        del learned_embed1,learned_embed2
    # Add learned concept

    from contextnet import ContextNet
    cls_net=ContextNet(768, len(token_embeds)-1)


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
    prior_concepts=[args.prior_concept1,args.prior_concept2]
    placeholder_tokens=[args.placeholder_token1,args.placeholder_token2]
    placeholder_ids=[placeholder_token_id1[0],placeholder_token_id2[0]]
    train_dataset_img = TextualInversionDatasetMulti(
        include_prior_concept=args.include_prior_concept,
        data_root1=args.train_data_dir1,
        data_root2=args.train_data_dir2,
        tokenizer=tokenizer,
        size=args.resolution,
        repeats=args.repeats,
        learnable_property=args.learnable_property,
        center_crop=args.center_crop,
        flip_p=args.flip_p,
        exclude_suffix=args.exclude_suffix,
        mask_token_ids=mask_token_ids[0],
        mlm_target=args.mlm_target,
        get_images=True,
        prompt_type=args.prompt_type,
        # multi
        placeholder_tokens=placeholder_tokens,
        placeholder_ids=placeholder_ids,
        prior_concepts=prior_concepts,
        make_composition=args.make_composition,
    )
    train_dataset_mlm_multi = TextualInversionDatasetMulti(
        include_prior_concept=args.include_prior_concept,
        data_root1=args.train_data_dir1,
        data_root2=args.train_data_dir2,
        tokenizer=tokenizer,
        size=args.resolution,
        repeats=args.repeats,
        learnable_property=args.learnable_property,
        center_crop=args.center_crop,
        flip_p=args.flip_p,
        exclude_suffix=args.exclude_suffix,
        mask_token_ids=mask_token_ids[0],
        mlm_target=args.mlm_target,
        get_images=False,
        prompt_type=args.prompt_type,
        placeholder_tokens=placeholder_tokens,
        placeholder_ids=placeholder_ids,
        prior_concepts=prior_concepts,
        make_composition=args.make_composition,
    )
   
    
    def collate_fn(examples):
        if 'pixel_values' in examples[0]:
            # 1. pixel_values
            pixel_values = [example["pixel_values"] for example in examples]
            pixel_values = torch.stack(pixel_values)
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

            # 2. input_ids
            input_ids = [example["input_ids"] for example in examples]
            input_ids=torch.stack(input_ids)
            # 2. input_ids
            is_keyword_tokens1 = [example["is_keyword_tokens1"] for example in examples] #N,77, list of booleans
            is_keyword_tokens2 = [example["is_keyword_tokens2"] for example in examples] #N,77, list of booleans
            is_keyword_tokens1 = torch.stack(is_keyword_tokens1)
            is_keyword_tokens2 = torch.stack(is_keyword_tokens2)

            masks = [example["masks"] for example in examples]
            masks = torch.stack(masks)
            masks = masks.to(memory_format=torch.contiguous_format).float()
        else:
            pixel_values=[]
            input_ids=[]
            is_keyword_tokens1=[]
            is_keyword_tokens2=[]
            masks=[]

       
        # 3. For MLM 
        input_ids_masked = [example["input_ids_masked"] for example in examples]
        input_ids_masked=torch.stack(input_ids_masked)
        input_ids_pos = [example["input_ids_pos"] for example in examples]

        
        input_ids_pos=torch.stack(input_ids_pos)
        masked_idxs = [example["masked_idxs"] for example in examples] #N,77, list of booleans
        masked_idxs = torch.stack(masked_idxs)
        mlm_labels = [example["mlm_labels"] for example in examples] #N,77, list of booleans
        mlm_labels = torch.stack(mlm_labels)
        non_special_idxs = [example["non_special_idxs"] for example in examples] #N,77, list of booleans
        non_special_idxs = torch.stack(non_special_idxs)

        
        # for triplet
        is_keyword_tokens_mlm1 = [example["is_keyword_tokens_mlm1"] for example in examples] #N,77, list of booleans
        is_keyword_tokens_mlm2 = [example["is_keyword_tokens_mlm2"] for example in examples] #N,77, list of booleans

        is_keyword_tokens_mlm1 = torch.stack(is_keyword_tokens_mlm1)
        is_keyword_tokens_mlm2 = torch.stack(is_keyword_tokens_mlm2)

        input_ids_single1 = [example["input_ids_single1"] for example in examples]
        input_ids_single2 = [example["input_ids_single2"] for example in examples]
        input_ids_single1=torch.stack(input_ids_single1)
        input_ids_single2=torch.stack(input_ids_single2)

        is_keyword_tokens_single1 = [example["is_keyword_tokens_single1"] for example in examples] #N,77, list of booleans
        is_keyword_tokens_single2 = [example["is_keyword_tokens_single2"] for example in examples] #N,77, list of booleans
        is_keyword_tokens_single1 = torch.stack(is_keyword_tokens_single1)
        is_keyword_tokens_single2 = torch.stack(is_keyword_tokens_single2)
        # for triplet

        
        # 3. For MLM 


        batch = {
            "pixel_values": pixel_values,
            "input_ids": input_ids, # for reconstruction

            "input_ids_masked": input_ids_masked, # for mlm
            "input_ids_pos": input_ids_pos, # for mlm

            "masked_idxs": masked_idxs,
            "mlm_labels": mlm_labels,

            "non_special_idxs": non_special_idxs,
            "is_keyword_tokens1": is_keyword_tokens1, # for reconstruction
            "is_keyword_tokens2": is_keyword_tokens2, # for reconstruction
            "input_ids_single1": input_ids_single1, # for triplet
            "input_ids_single2": input_ids_single2, # for triplet
            "is_keyword_tokens_mlm1": is_keyword_tokens_mlm1,# for triplet
            "is_keyword_tokens_mlm2": is_keyword_tokens_mlm2,# for triplet


            "is_keyword_tokens_single1": is_keyword_tokens_single1, # for triplet
            "is_keyword_tokens_single2": is_keyword_tokens_single2, # for triplet

            "masks": masks,
        }
        return batch
    train_dataloader_img = torch.utils.data.DataLoader(
        train_dataset_img, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
    )
    train_dataloader_mlm_multi = torch.utils.data.DataLoader(
        train_dataset_mlm_multi, batch_size=args.mlm_batch_size, shuffle=True, num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
    )
    

    mlm_loader_multi = cycle(train_dataloader_mlm_multi)
    def load_mlm_batch(mlm_loader):
        mlm_data_multi=next(mlm_loader)
        return mlm_data_multi


    if args.validation_epochs is not None:
        warnings.warn(
            f"FutureWarning: You are doing logging with validation_epochs={args.validation_epochs}."
            " Deprecated validation_epochs in favor of `validation_steps`"
            f"Setting `args.validation_steps` to {args.validation_epochs * len(train_dataset_img)}",
            FutureWarning,
            stacklevel=2,
        )
        args.validation_steps = args.validation_epochs * len(train_dataset_img)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader_mlm_multi) / args.gradient_accumulation_steps)
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
    
    text_encoder, optimizer, train_dataloader_img, lr_scheduler,cls_net,mlm_loader_multi = accelerator.prepare(
        text_encoder, optimizer, train_dataloader_img, lr_scheduler,cls_net,mlm_loader_multi
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
    num_update_steps_per_epoch = math.ceil(len(train_dataloader_img) / args.gradient_accumulation_steps)
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

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader_mlm_multi)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=0,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    # keep original embeddings as reference
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=unet,
        vae=vae,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=False)

    orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()
    cos_sim=torch.nn.CosineSimilarity(dim=-1, eps=1e-08)
    from supconloss import SupervisedContrastiveLoss
    contrastive_criterion=SupervisedContrastiveLoss()
    import time
    for epoch in range(first_epoch, args.num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader_img):
            with accelerator.accumulate(text_encoder):
                # 1. Load Batch
                pixel_values=batch["pixel_values"]# B,77 list of booleans (tensor)
                input_ids=batch["input_ids"]# B,77 list of booleans (tensor)
                masks=batch["masks"]# B,77 list of booleans (tensor)
                is_keyword_tokens1=batch["is_keyword_tokens1"]# B,77 list of booleans (tensor)
                is_keyword_tokens2=batch["is_keyword_tokens2"]# B,77 list of booleans (tensor)
                mask64=torch.nn.functional.interpolate(masks,(64,64),mode='nearest')
                
                # for multi MLM
                batch_mlm_multi=load_mlm_batch(mlm_loader_multi)
                input_ids_pos=batch_mlm_multi["input_ids_pos"].to(accelerator.device)# B,77 list of booleans (tensor)
                masked_idxs=batch_mlm_multi["masked_idxs"]
                mlm_labels=batch_mlm_multi["mlm_labels"].to(accelerator.device)
                non_special_idxs=batch_mlm_multi["non_special_idxs"]
                input_ids_masked=batch_mlm_multi["input_ids_masked"].to(accelerator.device)
                input_ids_single1=batch_mlm_multi["input_ids_single1"].to(accelerator.device)
                input_ids_single2=batch_mlm_multi["input_ids_single2"].to(accelerator.device)
                is_keyword_tokens_single1=batch_mlm_multi["is_keyword_tokens_single1"].to(accelerator.device)
                is_keyword_tokens_single2=batch_mlm_multi["is_keyword_tokens_single2"].to(accelerator.device)
                is_keyword_tokens_mlm1=batch_mlm_multi["is_keyword_tokens_mlm1"].to(accelerator.device)
                is_keyword_tokens_mlm2=batch_mlm_multi["is_keyword_tokens_mlm2"].to(accelerator.device)
                # for multi MLM
                # 1. Load Batch

                

                
                
                # 2. Reconstruction Loss
                latents = vae.encode(pixel_values.to(dtype=weight_dtype)).latent_dist.sample().detach()
                latents = latents * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                learned_embed1=accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[min(placeholder_token_id1) : max(placeholder_token_id1) + 1]
                learned_embed2=accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[min(placeholder_token_id2) : max(placeholder_token_id2) + 1]
                if args.normalize_target1:
                    target_emb1=F.normalize(learned_embed1,p=1,dim=-1)*args.normalize_target1
                else:
                    target_emb1=learned_embed1
                if args.normalize_target2:
                    target_emb2=F.normalize(learned_embed2,p=1,dim=-1)*args.normalize_target2
                else:
                    target_emb2=learned_embed2
                encoder_hidden_states = text_encoder(input_ids,
                                        is_keyword_tokens1=is_keyword_tokens1,
                                        inj_embeddings1=target_emb1,
                                        is_keyword_tokens2=is_keyword_tokens2,
                                        inj_embeddings2=target_emb2,
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
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                if args.masked_loss:
                    loss=loss*mask64
                loss=loss.mean()
                # 2. Reconstruction Loss

                
                

                # 3. MLM Loss
                loss_mlm=None
                if args.lambda_mlm:
                    # last_hidden_state: torch.FloatTensor = None
                    # pooler_output: torch.FloatTensor = None
                    # hidden_states: Optional[Tuple[torch.FloatTensor]] = None
                    # attentions: Optional[Tuple[torch.FloatTensor]] = None
                    # keywords_similarities: Optional[torch.FloatTensor] = None
                    # nonkey_similarities: Optional[torch.FloatTensor] = None
                    text_encodings_mlm = text_encoder(input_ids_masked,
                                                  mask_embedding=mask_embeds.unsqueeze(0),
                                                  mask_idxs=masked_idxs,
                                                  is_keyword_tokens1=is_keyword_tokens_mlm1,
                                                  is_keyword_tokens2=is_keyword_tokens_mlm2,
                                                  inj_embeddings1=target_emb1,
                                                  inj_embeddings2=target_emb2,
                                                  output_hidden_states=True)#.to(dtype=weight_dtype)
                    clip_text_embedding_mlm=text_encodings_mlm.last_hidden_state.to(accelerator.device)
                    mlm_logits=cls_net(clip_text_embedding_mlm)
                    masked_idxs_flat=masked_idxs.view(-1)
                    loss_mlm = F.cross_entropy(
                        mlm_logits.view(-1,len(orig_embeds_params)-1),
                        mlm_labels.view(-1),
                        ignore_index=-100,
                        reduction='none'
                    )
                    loss_mlm[masked_idxs_flat]*=args.mlm_weight
                    loss_mlm=loss_mlm.mean()
                    loss+=(loss_mlm*args.lambda_mlm)
                loss_sim=None
                loss_sim_log=None
                if args.lambda_sim:
                    hidden_states_list=text_encodings_mlm.hidden_states
                    if args.dissim_layers=='last':
                        dissim_layers=[len(hidden_states_list)-1] 
                    elif args.dissim_layers=='all':
                        dissim_layers=np.arange(len(hidden_states_list)).tolist()
                    elif 'to' in args.dissim_layers:
                        first=int(args.dissim_layers.split('to')[0])
                        last=int(args.dissim_layers.split('to')[1])
                        dissim_layers=np.arange(len(hidden_states_list))[first:last+1]
                    else:
                        dissim_layers=np.array(args.dissim_layers.split(',')).astype(np.int32)
                    simlist=[]
                    for hidx,hidden_states in enumerate(hidden_states_list):
                        if hidx not in dissim_layers:
                            continue
                        hidden_states=hidden_states.to(accelerator.device)
                        key_embs1=hidden_states[is_keyword_tokens_mlm1]
                        key_embs2=hidden_states[is_keyword_tokens_mlm2]
                        sim=pairwise_cosine_similarity(key_embs1,key_embs2,reduction='mean')
                        simlist.append(sim)
                    # learned_embed1=accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id1].unsqueeze(0)
                    # learned_embed2=accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id2].unsqueeze(0)
                    # loss_sim=1-cos_sim(learned_embed2,learned_embed1).mean()
                    simlist=torch.cat(simlist)
                    loss_sim_log=simlist.detach().abs().mean()
                    if args.sim_margin:
                        simlist_margin=torch.clip(simlist.abs()-args.sim_margin,min=0)
                        if torch.sum(simlist_margin>0):
                            loss_sim=simlist_margin[torch.where(simlist_margin>0)].mean()    
                            loss+=(loss_sim*args.lambda_sim)
                        else:
                            print('here')
                    else:
                        loss_sim=simlist.abs().mean() 
                        loss+=(loss_sim*args.lambda_sim)
                loss_contrastive=None
                if args.lambda_contrastive:
                    clip_text_single1 = text_encoder(input_ids_single1,
                                                     is_keyword_tokens1=is_keyword_tokens_single1,
                                                    inj_embeddings1=target_emb1,
                    )[0].to(accelerator.device, dtype=weight_dtype)
                    clip_text_single2 = text_encoder(input_ids_single2,
                                                     is_keyword_tokens2=is_keyword_tokens_single2,
                                                    inj_embeddings2=target_emb2,
                    )[0].to(accelerator.device, dtype=weight_dtype)
                    if not args.lambda_mlm:
                        text_encodings = text_encoder(input_ids_masked,
                                                  mask_embedding=mask_embeds.unsqueeze(0),
                                                  mask_idxs=masked_idxs,
                                                  is_keyword_tokens1=is_keyword_tokens_mlm1,
                                                  is_keyword_tokens2=is_keyword_tokens_mlm2,
                                                  inj_embeddings1=target_emb1,
                                                  inj_embeddings2=target_emb2,
                                                  output_attentions=True)#.to(dtype=weight_dtype)
                        clip_text_embedding_mlm=text_encodings[0].to(accelerator.device)
                    multi_concept1=clip_text_embedding_mlm[is_keyword_tokens_mlm1]
                    multi_concept2=clip_text_embedding_mlm[is_keyword_tokens_mlm2]
                    single_concept1=clip_text_single1[is_keyword_tokens_single1]
                    single_concept2=clip_text_single2[is_keyword_tokens_single2]
                    features=torch.cat([multi_concept1,multi_concept2,single_concept1,single_concept2])
                    features=F.normalize(features,p=2,dim=-1)

                    target1=torch.zeros(len(multi_concept1))
                    target2=torch.ones(len(multi_concept2))
                    target3=torch.zeros(len(single_concept1))
                    target4=torch.ones(len(single_concept2))
                    target=torch.cat([target1,target2,target3,target4])
                    loss_contrastive=contrastive_criterion(features,target)
                    loss+=(loss_contrastive*args.lambda_contrastive)

                    

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # Let's make sure we don't update any embedding weights besides the newly added token
                index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
                assert isinstance(mask_token_ids,list)
                assert isinstance(placeholder_token_id1,list)
                assert isinstance(placeholder_token_id2,list)
                if args.freeze_mask_embedding:
                    # if freeze mask
                    # do not include in no_updates list
                    index_no_updates[min(placeholder_token_id1+placeholder_token_id2) : max(placeholder_token_id1+placeholder_token_id2) + 1] = False
                else:
                    index_no_updates[min(placeholder_token_id1+placeholder_token_id2+mask_token_ids) : max(placeholder_token_id1+placeholder_token_id2+mask_token_ids) + 1] = False
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
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(ckpt_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)
                    learned_embed1=accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id1]
                    learned_embed2=accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id2]
                    learned_embeds_dict = {args.placeholder_token1: learned_embed1.detach().cpu(),
                                           args.placeholder_token2: learned_embed2.detach().cpu()}
                    weight_name = f"learned_embeds_multi_s{global_step}.pt"
                    save_path = os.path.join(ckpt_dir, weight_name)
                    torch.save(learned_embeds_dict, save_path)
                if ((global_step % args.validation_steps == 0)):
                    # visualize input
                    input_image=(pixel_values[0].permute(1,2,0).detach().cpu().numpy()+1)*127.5
                    input_mask=masks[0].permute(1,2,0).detach().cpu().numpy()
                    if args.masked_loss:
                        input_image=input_image*input_mask
                        input_image=input_image.astype(np.uint8)
                    else:
                        input_image=input_image.astype(np.uint8)
                    input_image=Image.fromarray(input_image)
                    input_image.save(os.path.join(viz_dir,'input_s{:05d}.jpg'.format(global_step)))
                    if args.lambda_mlm:
                        # 1. MLM Result Logging
                        is_keyword_tokens_mlm1=is_keyword_tokens_mlm1
                        is_keyword_tokens_mlm2=is_keyword_tokens_mlm2
                        input_ids_key1=input_ids_pos[is_keyword_tokens_mlm1][:5]
                        input_ids_key2=input_ids_pos[is_keyword_tokens_mlm2][:5]
                        decoded_key1=tokenizer.batch_decode(input_ids_key1)
                        decoded_key2=tokenizer.batch_decode(input_ids_key2)
                        decoded_key1_list=[]
                        decoded_key2_list=[]
                        for dec1 in decoded_key1:
                            decoded_key1_list.append('{:8}'.format(dec1))
                        for dec2 in decoded_key2:
                            decoded_key2_list.append('{:8}'.format(dec2))
                        decoded_key1=' '.join(decoded_key1_list)
                        decoded_key2=' '.join(decoded_key2_list)
                        viz_idx=0
                        masked_idxs=masked_idxs.detach().cpu().numpy()[viz_idx:viz_idx+1]
                        non_special_idxs=non_special_idxs.detach().cpu()[viz_idx:viz_idx+1]
                        mlm_logits=mlm_logits.argmax(-1).detach().cpu().numpy()[viz_idx:viz_idx+1]#1,77
                        input_ids_pos=input_ids_pos[viz_idx:viz_idx+1]
                        
                        input_ids_masked=input_ids_masked[viz_idx:viz_idx+1]
                        input_ids_pos=input_ids_pos[non_special_idxs]
                        input_ids_masked=input_ids_masked[non_special_idxs]
                        mlm_logits=mlm_logits[non_special_idxs]
                        masked_idxs=masked_idxs[non_special_idxs]
                        decoded=tokenizer.batch_decode(input_ids_pos)
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
                        dots='-'*150
                        print()
                        print()
                        print(dots)
                        print(dots)
                        print('Step\t\t|{}'.format(global_step))
                        print('Raw\t\t|{}'.format(decoded))
                        print('Masked\t\t|{}'.format(decoded_masked))
                        print('Preds\t\t|{}'.format(decoded_logits))
                        print(dots)
                        print('Key1\t\t|{}'.format(decoded_key1))
                        print('Key2\t\t|{}'.format(decoded_key2))
                        print(dots)
                        print()
                        # 1. MLM Result Logging
                    if not args.debug:
                        # 2. Image Logging
                        with torch.no_grad():
                            if args.normalize_target1:
                                target_emb1=F.normalize(learned_embed1,p=1,dim=-1)*args.normalize_target1
                            else:
                                target_emb1=learned_embed1
                            if args.normalize_target2:
                                target_emb2=F.normalize(learned_embed2,p=1,dim=-1)*args.normalize_target2
                            else:
                                target_emb2=learned_embed2
                            print(target_emb1.shape,'target_emb1.shape')
                            print(target_emb2.shape,'target_emb2.shape')
                            images,validation_prompts = log_validation(
                                tokenizer=tokenizer, 
                                args=args,
                                accelerator=accelerator,
                                target_emb1=target_emb1,
                                target_emb2=target_emb2,
                                pipeline=pipeline,

                            )
                        # save images
                        # validation_files=os.listdir(args.train_data_dir)
                        validation_files1=sorted(os.listdir(args.train_data_dir1))
                        validation_files2=sorted(os.listdir(args.train_data_dir2))
                        validation_target1=Image.open(os.path.join((args.train_data_dir1),validation_files1[0])).resize((512,512)).convert('RGB')
                        validation_target2=Image.open(os.path.join((args.train_data_dir2),validation_files2[0])).resize((512,512)).convert('RGB')


                        num_images=len(images)
                        num_cols=num_images
                        num_rows=num_images//num_cols
                        margin_bottom=150
                        margin_right=10
                        merged_viz = Image.new('RGB', ((512+margin_right)*(num_cols+2), (512+margin_bottom)*num_rows), (255, 255, 255))
                        for ridx in range(num_rows):
                            merged_viz.paste(validation_target1,(0,ridx*(512+margin_bottom)))
                            merged_viz.paste(validation_target2,((512+margin_right),ridx*(512+margin_bottom))) 
                        for iidx,(image, val_prompt) in enumerate(zip(images[:],validation_prompts[:])):
                            row_idx=iidx//num_cols
                            col_idx=iidx-(num_cols*row_idx)
                            x0=(col_idx+2)*(512+margin_right)
                            y0=row_idx*(512+margin_bottom)+512
                            x1=x0+(512+margin_right)
                            y1=y0+margin_bottom
                            merged_viz=render_caption(merged_viz,val_prompt,[x0,y0+20,x1,y1])
                            merged_viz.paste(image.convert('RGB'),((col_idx+2)*(512+margin_right),row_idx*(512+margin_bottom)))
                        merged_viz.save(os.path.join(sample_dir, 'sample_{:05d}.jpg'.format(global_step)))
                        if args.report_to=='wandb':
                            if (global_step) % args.save_steps == 0:   
                                wandb_image = wandb.Image(merged_viz, caption="img_{:06d}_result.jpg".format(global_step))
                                run.log({"examples": wandb_image})
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            with torch.no_grad():
                norm_target1=torch.norm(target_emb1,p=1,dim=-1)
                norm_target2=torch.norm(target_emb2,p=1,dim=-1)
                # print(norm_target1.item(),'norm_target1')
            if loss_sim_log is not None:
                logs['loss_sim']=loss_sim_log.detach().item()#*args.lambda3
            if loss_mlm is not None:
                logs['loss_mlm']=loss_mlm.detach().item()#*args.lambda3
            if loss_contrastive is not None:
                logs['loss_contrastive']=loss_contrastive.detach().item()#*args.lambda3
            # if loss_triplet is not None:
            #     logs['loss_triplet']=loss_triplet.detach().item()
            #     logs['sim_pos']=torch.mean(sim_pos).detach().item()
            #     logs['sim_neg']=torch.mean(sim_neg).detach().item()
            logs['norm_target1']=norm_target1.item()
            logs['norm_target2']=norm_target2.item()
            if args.report_to=='wandb' and accelerator.is_main_process:
                wandb.log(logs)
            if args.silent:
                log_keys=list(logs.keys())
                for lk in log_keys:
                    print('{}:{:.4f}\t'.format(lk,logs[lk]))
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
        weight_name = "learned_embeds_multi_final.pt" 
        save_path = os.path.join(ckpt_dir, weight_name)
        # learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
        # learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}
        learned_embed1=accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id1]
        learned_embed2=accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id2]
        learned_embeds_dict = {args.placeholder_token1: learned_embed1.detach().cpu(),
                                args.placeholder_token2: learned_embed2.detach().cpu()}
        torch.save(learned_embeds_dict, save_path)

        

    accelerator.end_training()


if __name__ == "__main__":
    main()