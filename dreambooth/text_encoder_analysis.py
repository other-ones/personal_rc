from utils import float_to_str
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
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
    StableDiffusionPipelineConcept,
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

from datasets_pkgs.dataset_analysis_multi import TextualInversionDatasetMulti

# ------------------------------------------------------------------------------


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.28.0.dev0")

logger = get_logger(__name__)






def main():
    args = parse_args()
    # dict_args=vars(args)
    exp_name=args.resume_unet_path.split('/')[-4]
    exp_dir=os.path.join(args.output_dir,exp_name)  
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
    # vae = AutoencoderKL.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    # )
    # unet = UNet2DConditionModel.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    # )

    # HERE
    mask_tokens = [args.mask_tokens]
    placeholder_token1 = [args.placeholder_token1]
    placeholder_token2 = [args.placeholder_token2]
    tokenizer.add_tokens(mask_tokens)
    tokenizer.add_tokens(placeholder_token1)
    tokenizer.add_tokens(placeholder_token2)
    text_encoder.resize_token_embeddings(len(tokenizer))
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
    mask_token_ids = tokenizer.convert_tokens_to_ids(mask_tokens)
    placeholder_token_id1 = tokenizer.convert_tokens_to_ids(placeholder_token1)
    placeholder_token_id2 = tokenizer.convert_tokens_to_ids(placeholder_token2)
    # HERE





    
    
    

    # if args.resume_unet_path and args.resume_unet_path!='None':
    #     state_dict = torch.load(args.resume_unet_path, map_location=torch.device('cpu'))
    #     if not isinstance(state_dict,OrderedDict):
    #         state_dict=state_dict()
    #     unet.load_state_dict(state_dict,strict=True)
    #     print('unet parameters loaded')
    #     del state_dict
    # if args.resume_text_encoder_path and args.resume_text_encoder_path!='None':
    #     state_dict = torch.load(args.resume_text_encoder_path, map_location=torch.device('cpu'))
    #     if not isinstance(state_dict,OrderedDict):
    #         state_dict=state_dict()
    #     text_encoder.load_state_dict(state_dict,strict=True)
    #     print('text_encoder parameters loaded')
    #     del state_dict

    



    # # Freeze vae and unet
    # vae.requires_grad_(False)
    # unet.requires_grad_(False)
    # # Freeze all parameters except for the token embeddings in text encoder
    # text_encoder.text_model.encoder.requires_grad_(False)
    # text_encoder.text_model.final_layer_norm.requires_grad_(False)
    # text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
    # if args.gradient_checkpointing: #FAlse
    #     # Keep unet in train mode if we are using gradient checkpointing to save memory.
    #     # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
    #     unet.train()
    #     text_encoder.gradient_checkpointing_enable()
    #     unet.enable_gradient_checkpointing()
    # unet.eval()
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
    

    # Dataset and DataLoaders creation:
    prior_concepts=[args.prior_concept1,args.prior_concept2]
    placeholder_tokens=[args.placeholder_token1,args.placeholder_token2]
    placeholder_ids=[placeholder_token_id1[0],placeholder_token_id2[0]]
    
    train_dataset_mlm_multi = TextualInversionDatasetMulti(
        include_prior_concept=args.include_prior_concept,
        tokenizer=tokenizer,
        size=args.resolution,
        repeats=args.repeats,
        center_crop=args.center_crop,
        flip_p=args.flip_p,
        mask_token_ids=mask_token_ids[0],
        mlm_target=args.mlm_target,
        get_images=False,
        prompt_type=args.prompt_type,

        # multi
        data_root1=args.train_data_dir1,
        data_root2=args.train_data_dir2,
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
        is_keyword_tokens2 = [example["is_keyword_tokens2"] for example in examples] #N,77, list of booleans
        is_keyword_tokens2 = torch.stack(is_keyword_tokens2)

        is_prior1 = [example["is_prior1"] for example in examples] #N,77, list of booleans
        is_prior1 = torch.stack(is_prior1)
        is_prior2 = [example["is_prior2"] for example in examples] #N,77, list of booleans
        is_prior2 = torch.stack(is_prior2)
        # for contrast
        
        # 3. For MLM 


        batch = {
            "raw_captions": raw_captions,
            "input_ids_pos": input_ids_pos, # for mlm
            "non_special_idxs": non_special_idxs,
            "non_keyword_idxs": non_keyword_idxs,
            "is_keyword_tokens1": is_keyword_tokens1,# for triplet
            "is_keyword_tokens2": is_keyword_tokens2,# for triplet
            "is_prior1": is_prior1,# for triplet
            "is_prior2": is_prior2,# for triplet
        }
        return batch
    train_dataloader_mlm_multi = torch.utils.data.DataLoader(
        train_dataset_mlm_multi, batch_size=args.mlm_batch_size, shuffle=True, num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
    )
    



    

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader_mlm_multi) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True


    text_encoder.train()
    text_encoder, train_dataloader_mlm_multi = accelerator.prepare(
        text_encoder, train_dataloader_mlm_multi
    )
    

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # # Move vae and unet to device and cast to weight_dtype
    # unet.to(accelerator.device, dtype=weight_dtype)
    # vae.to(accelerator.device, dtype=weight_dtype)

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
    # Potentially load in the weights and states from a previous save

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
    # pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    # pipeline = pipeline.to(accelerator.device)
    # pipeline.set_progress_bar_config(disable=False)

    import time
    text_encoder.train()
    for step, batch_text_multi in enumerate(train_dataloader_mlm_multi):
        # for multi MLM
        input_ids_pos=batch_text_multi["input_ids_pos"]# B,77 list of booleans (tensor)
        raw_captions=batch_text_multi["raw_captions"]
        non_special_idxs=batch_text_multi["non_special_idxs"]
        non_keyword_idxs=batch_text_multi["non_keyword_idxs"]
        is_keyword_tokens1=batch_text_multi["is_keyword_tokens1"].to(accelerator.device)
        is_keyword_tokens2=batch_text_multi["is_keyword_tokens2"].to(accelerator.device)
        is_prior1=batch_text_multi["is_prior1"].to(accelerator.device)
        is_prior2=batch_text_multi["is_prior2"].to(accelerator.device)
        # for multi MLM
        input_ids_key1=input_ids_pos[is_keyword_tokens1]
        input_ids_key2=input_ids_pos[is_keyword_tokens2]
        decoded_key1=tokenizer.batch_decode(input_ids_key1)
        decoded_key1_list=[]
        decoded_key2=tokenizer.batch_decode(input_ids_key2)
        decoded_key2_list=[]




        # Target Encodings
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
        if accelerator.is_main_process:
            count=1
            print(args.calibrate_kneg,'calibrate_kneg')
            print(args.calibrate_kpos,'calibrate_kpos')
            print(args.calibrate_pneg,'calibrate_pneg')
            print(args.calibrate_ppos,'calibrate_ppos')
            out = text_encoder(input_ids_pos,
                                is_keyword_tokens1=is_keyword_tokens1,
                                inj_embeddings1=target_emb1,
                                is_keyword_tokens2=is_keyword_tokens2,
                                inj_embeddings2=target_emb2,
                                output_similarities=True,
                                output_attentions=True,
                                non_keyword_idxs=non_keyword_idxs,
                                calibrate_kneg=args.calibrate_kneg,
                                calibrate_kpos=args.calibrate_kpos,
                                calibrate_pneg=args.calibrate_pneg,
                                calibrate_ppos=args.calibrate_ppos,
                                is_prior1=is_prior1,
                                is_prior2=is_prior2,
                                )
            # is_keyword_tokens1 # 400,77
            attention_per_layers=out.attentions #[12,12,400,77,77]
            print(len(attention_per_layers),'len(attention_per_layers)') #12
            print(is_keyword_tokens1.shape,'is_keyword_tokens1.shape') #400,77
            for layer_idx in range(len(attention_per_layers)):
                layer_attentions=attention_per_layers[layer_idx]
                # print(layer_idx,layer_attentions.shape,'layer_attentions.shape') # torch.Size([400, num_head, 77, 77])
                layer_attentions=torch.mean(layer_attentions,dim=1) # 400,77,77
                key1_attentions=layer_attentions[is_keyword_tokens1] # 400,77
                max_idx=torch.argmax(key1_attentions[:,1:],1)+1 # 400,
                key1_idx=torch.argmax(is_keyword_tokens1.int(),1) # 400,
                key2_idx=torch.argmax(is_keyword_tokens2.int(),1) # 400,
                key1_key1_attentions=key1_attentions[is_keyword_tokens1] # 400
                key1_key2_attentions=key1_attentions[is_keyword_tokens2] # 400
                diff=torch.abs(key1_key1_attentions-key1_key2_attentions) # 400
                diff_sum=diff.sum().item()
                diff_ratio=diff/key1_key1_attentions
                print(layer_idx,diff_ratio.mean(),'diff_ratio',torch.sum(max_idx==key2_idx).item()/len(key2_idx),'proportion2')
                # print(layer_idx,diff_sum,'diff_sum',torch.sum(max_idx==key2_idx).item()/len(key2_idx),'proportion2')
                
            keywords_similarities=out.keywords_similarities
            nonkey_similarities=out.nonkey_similarities
            nonkey_similarities=nonkey_similarities.detach().cpu().numpy()
            keywords_similarities=keywords_similarities.detach().cpu().numpy()
            xpoints=np.arange(13)
            for key_sims,nonkey_sims in zip(keywords_similarities,nonkey_similarities):
                plt.plot(xpoints,key_sims, 'b',linewidth=0.1)
                plt.plot(xpoints,nonkey_sims, 'r',linewidth=0.1)
                count+=1
            plt.savefig('simcurve.jpg',dpi=500)
            break
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()