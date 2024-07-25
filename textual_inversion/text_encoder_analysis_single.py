import matplotlib.pyplot as plt
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
# ADDED
if is_wandb_available():
    import wandb

from datasets_pkgs.dataset_analysis_single import TextualInversionDatasetSingle

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
            ]
    # vase
    else:
        assert False
    print(validation_prompts[0],'validation_prompts')
    print('Start Inference')
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
        is_keyword_tokens_list1.append(is_keyword_tokens1)
    is_keyword_tokens_list1=torch.stack(is_keyword_tokens_list1)
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)
    with autocast_ctx:
        images = pipeline(validation_prompts, num_inference_steps=25, generator=generator,
                          inj_embeddings1=target_emb1,
                          is_keyword_tokens1=is_keyword_tokens_list1,
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
    print(placeholder_token1,'placeholder_token1')
    tokenizer.add_tokens(mask_tokens)
    tokenizer.add_tokens(placeholder_token1)
    mask_token_ids = tokenizer.convert_tokens_to_ids(mask_tokens)
    placeholder_token_id1 = tokenizer.convert_tokens_to_ids(placeholder_token1)
    # initializer_token_id1 = tokenizer.encode(args.prior_concept1, add_special_tokens=False)
    # initializer_token_id1 = initializer_token_id1[0]
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    # with torch.no_grad():
    #     for token_id in placeholder_token_id1:
    #         token_embeds[token_id] = token_embeds[initializer_token_id1].clone()
    print(token_embeds.shape,'token_embeds.shape')
    
    
    # mask_embeds=token_embeds[mask_token_ids]
    if args.mask_embed_path is not None:
        mask_embeds=torch.load(args.mask_embed_path)[args.mask_tokens].to(accelerator.device)
        mask_embeds=F.normalize(mask_embeds,p=1,dim=-1)*args.avg_norm
    # Add the placeholder token in tokenizer


    # Add learned concept
    if args.learned_embed_path1:
        learned_embed1=torch.load(args.learned_embed_path1)#[args.placeholder_token]
        learned_embed1=learned_embed1[args.placeholder_token1]
        with torch.no_grad():
            token_embeds[placeholder_token_id1] = learned_embed1.clone()
        del learned_embed1
    # Add learned concept



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
    placeholder_tokens=[args.placeholder_token1]
    placeholder_ids=[placeholder_token_id1[0]]
    
    train_dataset_mlm_multi = TextualInversionDatasetSingle(
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
        

       
        # 3. For MLM 
        raw_captions = [example["raw_caption"] for example in examples]
        input_ids_pos = [example["input_ids_pos"] for example in examples]

        
        input_ids_pos=torch.stack(input_ids_pos)
        non_special_idxs = [example["non_special_idxs"] for example in examples] #N,77, list of booleans
        non_special_idxs = torch.stack(non_special_idxs)
        non_keyword_idxs = [example["non_keyword_idxs"] for example in examples] #N,77, list of booleans
        non_keyword_idxs = torch.stack(non_keyword_idxs)
        
        is_keyword_tokens1 = [example["is_keyword_tokens1"] for example in examples] #N,77, list of booleans
        is_keyword_tokens1 = torch.stack(is_keyword_tokens1)
        is_prior1 = [example["is_prior1"] for example in examples] #N,77, list of booleans
        is_prior1 = torch.stack(is_prior1)
        # 3. For MLM 


        batch = {
            "raw_captions": raw_captions,
            "input_ids_pos": input_ids_pos, # for mlm
            "non_special_idxs": non_special_idxs,
            "non_keyword_idxs": non_keyword_idxs,
            "is_keyword_tokens1": is_keyword_tokens1,# for triplet
            "is_prior1": is_prior1,# for triplet
        }
        return batch
        
    train_dataloader_mlm_multi = torch.utils.data.DataLoader(
        train_dataset_mlm_multi, 
        batch_size=args.mlm_batch_size, 
        shuffle=True, num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
    )
    



    

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
    
    text_encoder, optimizer, lr_scheduler,train_dataloader_mlm_multi = accelerator.prepare(
        text_encoder, optimizer, lr_scheduler,train_dataloader_mlm_multi
    )
    

    
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
    num_update_steps_per_epoch = math.ceil(len(train_dataloader_mlm_multi) / args.gradient_accumulation_steps)
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
    text_encoder.train()
    for step, batch_text_multi in enumerate(train_dataloader_mlm_multi):
        with accelerator.accumulate(text_encoder):
            # for multi MLM
            is_prior1=batch_text_multi["is_prior1"].to(accelerator.device)
            input_ids_pos=batch_text_multi["input_ids_pos"]# B,77 list of booleans (tensor)
            raw_captions=batch_text_multi["raw_captions"]
            non_special_idxs=batch_text_multi["non_special_idxs"]
            non_keyword_idxs=batch_text_multi["non_keyword_idxs"]
            is_keyword_tokens1=batch_text_multi["is_keyword_tokens1"].to(accelerator.device)
            # for multi MLM
            

            # 1. MLM Result Logging
            input_ids_key1=input_ids_pos[is_keyword_tokens1]
            decoded_key1=tokenizer.batch_decode(input_ids_key1)
            decoded_key1_list=[]
            num_logs=10
            print()
            print()
            dots='-'*100
            print(dots)
            print(dots)
            print('Step\t\t|{}'.format(global_step))
            print(dots)
            for dec1 in decoded_key1:
                    decoded_key1_list.append('{:8}'.format(dec1))
            decoded_key1=' '.join(decoded_key1_list[:num_logs])
            
            
            
            # print('Key2\t\t|{}'.format(decoded_key2))
            for viz_idx in range(num_logs):
                non_special_idxs_viz=non_special_idxs.detach().cpu()[viz_idx:viz_idx+1]
                input_ids_pos_viz=input_ids_pos[viz_idx:viz_idx+1]
                input_ids_pos_viz=input_ids_pos_viz[non_special_idxs_viz]
                decoded=tokenizer.batch_decode(input_ids_pos_viz)
                decoded_list=[]
                for dec in decoded:
                    decoded_list.append('{:8}'.format(dec))
                decoded=' '.join(decoded_list)
                print('Input\t\t|{}'.format(decoded))
            print(dots)
            print('Key1\t\t|{}'.format(decoded_key1))
            print(dots)
            print(dots)
            print()
            # 1. MLM Result Logging



            # Target Encodings
            learned_embed1=accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[min(placeholder_token_id1) : max(placeholder_token_id1) + 1]
            if args.normalize_target1:
                target_emb1=F.normalize(learned_embed1,p=1,dim=-1)*args.normalize_target1
            else:
                target_emb1=learned_embed1
            print(torch.sum(is_keyword_tokens1),len(is_keyword_tokens1))
            
            if accelerator.is_main_process:
                count=1
                # for iip in input_ids_pos:
                #     print(torch.sum(iip).sum(),'iip')
                out = text_encoder(input_ids_pos,
                                    is_keyword_tokens1=is_keyword_tokens1,
                                    inj_embeddings1=target_emb1,
                                    output_attentions=True,
                                    non_keyword_idxs=non_keyword_idxs,
                                    )
                attention_per_layers=out.attentions #[12,12,400,77,77]
                for layer_idx in range(len(attention_per_layers)):
                    layer_attentions=attention_per_layers[layer_idx]
                    layer_attentions=torch.mean(layer_attentions,dim=1) # 400,77,77
                    key1_attentions=layer_attentions[is_keyword_tokens1] # 400,77
                    key1_key1_attentions=key1_attentions[is_keyword_tokens1] # 400
                    key1_prior1_attentions=key1_attentions[is_prior1] # 400
                    print(key1_key1_attentions.shape,'key1_key1_attentions.shape')
                    print(key1_prior1_attentions.shape,'key1_prior1_attentions.shape')

                # xpoints=np.arange(13)
                # for key_sims,nonkey_sims in zip(keywords_similarities,nonkey_similarities):
                #     plt.plot(xpoints,key_sims, 'b',linewidth=0.1)
                #     plt.plot(xpoints,nonkey_sims, 'r',linewidth=0.1)
                #     print(key_sims[-1],nonkey_sims[-1])
                #     count+=1
                # plt.savefig('simcurve.jpg',dpi=500)
                break
            break
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()