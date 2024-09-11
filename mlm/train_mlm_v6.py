
import inspect
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
from torch.utils.data import Dataset
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from datasets_pkgs.dataset_mlm_text_v6 import MLMDataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, AutoProcessor
import diffusers
import sys

sys.path.insert(0, './packages')
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipelineConcept,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available

# Added
import json
from utils import numpy_to_pil
import cv2
from PIL import Image
import torchvision
from torch import nn
# Added




import wandb
def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]
    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                            (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)



# ------------------------------------------------------------------------------


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.28.0.dev0")
logger = get_logger(__name__)

from configs import parse_args
def main():
    args = parse_args()
    dict_args=vars(args)
    
    exp_dir=os.path.join(args.output_dir,args.run_name)    
    # logging_dir = os.path.join(exp_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=exp_dir, logging_dir=None)
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
    # 
    if accelerator.is_main_process:
        viz_dir = os.path.join(exp_dir,'viz')
        os.makedirs(viz_dir, exist_ok=True)
        codepath=os.path.join(exp_dir,'src')
        if os.path.exists(codepath) and 'tmp' not in codepath:
            assert False,'code exists'
        os.makedirs(codepath,exist_ok=True)
        
        os.system('cp *.py {}'.format(codepath))
        os.system('cp ../datasets_pkgs {} -R'.format(codepath))
        os.system('cp packages {} -R'.format(codepath))
        # copy clip
        os.makedirs(os.path.join(codepath,'clip_src'),exist_ok=True)
        target = os.readlink('clip_src/modeling_clip.py')
        shutil.copy2(target, '{}/clip_src/modeling_clip.py'.format(codepath))
        target = os.readlink('clip_src/modeling_outputs.py')
        shutil.copy2(target, '{}/clip_src/modeling_outputs.py'.format(codepath))
        # copy clip
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
        if exp_dir is not None:
            os.makedirs(exp_dir, exist_ok=True)
        sample_dir=os.path.join(exp_dir,'samples')
        ckpt_dir=os.path.join(exp_dir,'checkpoints')
        os.makedirs(ckpt_dir,exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    
    # Load tokenizer
    if args.tokenizer_name: #  NO
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")








    # Add Mask Token
    mask_tokens = [args.mask_tokens]
    # if args.num_vectors_general < 1 or args.num_vectors_personal<1:
    #     raise ValueError(f"--num_vectors has to be larger or equal to 1, but is {args.num_vectors}")
    # assert args.num_vectors_general==1
    print(len(tokenizer),'len(tokenizer)1')
    num_added_tokens = tokenizer.add_tokens(mask_tokens)
    # Convert the initializer_token, placeholder_token to ids
    mask_token_ids = tokenizer.convert_tokens_to_ids(mask_tokens)
    text_encoder.resize_token_embeddings(len(tokenizer))
    # Initialise the newly added general/personal token with the embeddings of the initializer token (for validation)
    token_embeds = text_encoder.get_input_embeddings().weight.data
    print(len(tokenizer),'len(tokenizer)2')
    print(mask_token_ids,'mask_token_ids')
    # Track text embeddings - for scale investigation
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

    # Normalize -> Scale mask embeds
    norms = [i.norm(p=1,dim=-1).item() for i in text_encoder.get_input_embeddings().weight.data.clone()]
    avg_norm = np.mean(norms)
    max_norm=np.max(norms)
    min_norm=np.min(norms)
    print(avg_norm,'avg_norm',min_norm,'min_norm',max_norm,'max_norm')
    # 8.525119902624288 avg_norm 2.9221171189419692e-06 min_norm 619.1633911132812 max_norm
    mask_embeds=token_embeds[mask_token_ids]
    mask_embeds=F.normalize(mask_embeds,p=1,dim=-1)*avg_norm
    with torch.no_grad():
        for token_id in mask_token_ids:
            token_embeds[token_id] = mask_embeds.clone()
    mask_embeds_copy=mask_embeds.detach().clone().to(accelerator.device)
    # mask_embeds_copy=torch.ones_like(mask_embeds).to(accelerator.device)
    mask_embeds=mask_embeds.to(accelerator.device)
    # For verification
    sample_text='a sign neuripsdfs {}'.format(args.mask_tokens*3)
    print(sample_text,'sample_text')
    sample_encoded_ids=tokenizer(
            sample_text,
            add_special_tokens=False,
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
    decoded=tokenizer.batch_decode(sample_encoded_ids)

    print(' '.join(decoded),'encoded->decoded')

    
        
    # from token_cls import TokenCLS
    # cls_net = TokenCLS(input_dim=768, output_dim=len(token_embeds))
    from contextnet_v3 import ContextNetV3
    if 'v1-5' in args.pretrained_model_name_or_path:
        embed_dim=768
    elif '-2-1' in args.pretrained_model_name_or_path:
        embed_dim=1024

    cls_net=ContextNetV3(embed_dim, len(token_embeds))
    num_hidden_layers=4
    in_proj_std = (embed_dim**-0.5) * ((2 * num_hidden_layers) ** -0.5)
    out_proj_std = (embed_dim**-0.5) 
    final_std=768**-0.5
    for key,val in cls_net.named_parameters():
        print(key,'key clsnet')
        if 'bias' in key:
            val.data.zero_()
        else:
            if 'layernorm' in key:
                val.data.fill_(1.0)
            if 'position' in key:
                val.data.normal_(mean=0.0, std=0.02)
            if 'in_proj' in key:
                val.data.normal_(mean=0.0, std=in_proj_std)
            if 'out_proj' in key:
                val.data.normal_(mean=0.0, std=out_proj_std)
            if 'final' in key:
                val.data.normal_(mean=0.0, std=final_std)
    # final.weight key clsnet
    # final.bias key clsnet
    # position_embedding.weight key clsnet
    # position_embedding.weight key
    # ca_layers.0.in_proj_weight key clsnet
    # ca_layers.0.in_proj_bias key clsnet
    # ca_layers.0.out_proj.weight key clsnet
    # ca_layers.0.out_proj.bias key clsnet
    # ca_layers.1.in_proj_weight key clsnet
    # ca_layers.1.in_proj_bias key clsnet
    # ca_layers.1.out_proj.weight key clsnet
    # ca_layers.1.out_proj.bias key clsnet
    # ca_layers.2.in_proj_weight key clsnet
    # ca_layers.2.in_proj_bias key clsnet
    # ca_layers.2.out_proj.weight key clsnet
    # ca_layers.2.out_proj.bias key clsnet
    # ca_layers.3.in_proj_weight key clsnet
    # ca_layers.3.in_proj_bias key clsnet
    # ca_layers.3.out_proj.weight key clsnet
    # ca_layers.3.out_proj.bias key clsnet
    
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
    params_to_optimize = [
        {"params": text_encoder.get_input_embeddings().parameters(), "lr": args.learning_rate},
        {"params": cls_net.parameters(), "lr": args.learning_rate},
        # {"params": mask_embeds, "lr": args.learning_rate},
        # {"params": mapper_general.parameters(), "lr": args.learning_rate},
    ]
    optimizer = torch.optim.AdamW(
        params_to_optimize, 
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    train_dataset = MLMDataset(
        caption_path=args.caption_path,
        tokenizer=tokenizer,
        mask_tokens=args.mask_tokens,
        mlm_target=args.mlm_target,
        mask_token_ids=mask_token_ids[0],
        whole_word_mask=args.whole_word_mask,
    )
    def collate_fn(examples):
        # 1. input ids
        input_ids = [example["input_ids"] for example in examples]
        input_ids=torch.stack(input_ids)
        input_ids_masked = [example["input_ids_masked"] for example in examples]
        input_ids_masked=torch.stack(input_ids_masked)
        # 1. input ids

        # 2. masked_idxs
        masked_idxs = [example["masked_idxs"] for example in examples] #N,77, list of booleans
        masked_idxs = torch.stack(masked_idxs)
        # 2. masked_idxs




        # 3. mlm_labels
        mlm_labels = [example["mlm_labels"] for example in examples] #N,77, list of booleans
        mlm_labels = torch.stack(mlm_labels)
        # 3. mlm_labels

        non_special_idxs = [example["non_special_idxs"] for example in examples] #N,77, list of booleans
        non_special_idxs = torch.stack(non_special_idxs)


        
        

        batch = {
            "input_ids": input_ids,
            "input_ids_masked": input_ids_masked,
            "masked_idxs": masked_idxs,
            "mlm_labels": mlm_labels,
            "non_special_idxs": non_special_idxs,
        }
        return batch
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.train_batch_size, 
        shuffle=True, 
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn
    )

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
    print(len(train_dataloader),'len(train_dataloader)')
    text_encoder, optimizer, train_dataloader, lr_scheduler,cls_net = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler,cls_net
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    # Move vae and unet to device and cast to weight_dtype
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    # if accelerator.is_main_process:
    #     accelerator.init_trackers("textual_inversion", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0


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






    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=0,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    ce_criterion = torch.nn.CrossEntropyLoss()
    cos_sim=torch.nn.CosineSimilarity(dim=-1, eps=1e-08)
    import time
    inference_noise=None
    orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()
    print('start')
    for epoch in range(first_epoch, args.num_train_epochs):
        # text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            # Load Batch
            input_ids=batch["input_ids"]
            input_ids_masked=batch["input_ids_masked"]
            masked_idxs=batch["masked_idxs"]
            mlm_labels=batch["mlm_labels"]
            non_special_idxs=batch["non_special_idxs"]
            # Load Batch





            # clip-text
            bsz=len(input_ids)
            # mask_embeds=accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data[mask_token_ids]
            # if args.normalize_mask_embeds:
            #     mask_embeds_target=(F.normalize(mask_embeds,p=1,dim=1)*avg_norm).unsqueeze(0).to(accelerator.device)
            # else:
            #     mask_embeds_target=mask_embeds
            clip_text_embedding_masked = text_encoder(input_ids_masked,
                                            # mask_embedding=mask_embeds,
                                            # mask_idxs=masked_idxs
                                            )[0].to(accelerator.device, dtype=weight_dtype)
            # clip_text_embedding_masked = text_encoder(input_ids_masked,
            #                             normalizing_scale=avg_norm,
            #                             normalizing_idxs=mask_token_ids)[0].to(accelerator.device, dtype=weight_dtype)
            # print(clip_text_embedding_masked.shape,'clip_text_embedding_masked.shape')
            # torch.Size([150, 77, 768]) clip_text_embedding_masked.shape
            mlm_logits=cls_net(clip_text_embedding_masked)
            
            masked_idxs_flat=masked_idxs.view(-1)
            non_masked_idxs=~masked_idxs_flat
            
            loss_mlm = F.cross_entropy(
                mlm_logits.view(-1,len(orig_embeds_params)),
                mlm_labels.view(-1),
                ignore_index=-100,
                reduction='none'
            )
            loss_mlm[masked_idxs_flat]*=args.mlm_weight
            loss_mlm=loss_mlm.mean()
            # torch.Size([150, 77, 49409]) mlm_logits
            # torch.Size([150, 77]) masked_idxs
            # torch.Size([150, 77, 49409]) mlm_logits
            # torch.Size([150, 77]) masked_idxs
            # exit()
            # mlm_logits: n,l,vocab_size

            
            

            accelerator.backward(loss_mlm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
            index_no_updates[min(mask_token_ids) : max(mask_token_ids) + 1] = False
            with torch.no_grad():
                accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                    index_no_updates
                ] = orig_embeds_params[index_no_updates]


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # mapper_general.eval()
                if ((global_step)%args.visualize_steps)==0 and accelerator.is_main_process:
                    print(args.run_name)
                    viz_idx=1
                    masked_idxs=masked_idxs.detach().cpu().numpy()[viz_idx:viz_idx+1]
                    non_special_idxs=non_special_idxs.detach().cpu()[viz_idx:viz_idx+1]
                    mlm_logits=mlm_logits.argmax(-1).detach().cpu().numpy()[viz_idx:viz_idx+1]#1,77
                    input_ids=input_ids[viz_idx:viz_idx+1]
                    input_ids_masked=input_ids_masked[viz_idx:viz_idx+1]
                    
                    # TMP
                    # print(input_ids.shape,'input_ids.shape')
                    # print(input_ids_masked.shape,'input_ids_masked.shape')
                    # input_decoded=tokenizer.batch_decode(input_ids[0])
                    # masked_decoded=tokenizer.batch_decode(input_ids_masked[0])
                    
                    # print(np.array(input_decoded)[masked_idxs[0]],'masked_labels')
                    # print(np.array(masked_decoded)[masked_idxs[0]],'masked_decoded')
                    # TMP

                    input_ids=input_ids[non_special_idxs]
                    input_ids_masked=input_ids_masked[non_special_idxs]
                    mlm_logits=mlm_logits[non_special_idxs]
                    masked_idxs=masked_idxs[non_special_idxs]
              
                    decoded=tokenizer.batch_decode(input_ids)
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
                    
                        

                if (global_step) % (args.save_steps) == 0 or global_step==100:
                    if accelerator.is_main_process:
                        mask_embeds_saved=accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data[mask_token_ids].clone()
                        if global_step>-1 and (not args.debug):
                            if args.checkpoints_total_limit >0:
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
                            save_path=os.path.join(ckpt_dir,f'checkpoint-{global_step}')
                            os.makedirs(save_path,exist_ok=True)
                            save_path_cls_net = (f"{save_path}/cls_net_{global_step}_ckpt.pt")
                            save_path_mask_embeds = (f"{save_path}/mask_embeds_{global_step}_ckpt.pt")
                            save_path_optimizer = (f"{save_path}/optimizer_{global_step}_ckpt.pt")
                            print(checkpoints,'checkpoints')
                            print(args.checkpoints_total_limit,'checkpoints_total_limit')
                            print(f"save weights {save_path_cls_net}")
                            torch.save(cls_net.state_dict(), save_path_cls_net)
                            learned_embeds_dict = {args.mask_tokens: mask_embeds_saved.detach().cpu()}

                            print(f"save mask embeds {save_path_mask_embeds}",'cos_sim:{}'.format(cos_sim(mask_embeds_copy,mask_embeds_saved).item()))
                            torch.save(learned_embeds_dict, save_path_mask_embeds)
                            print(f"save optimizer {save_path_optimizer}")
                            torch.save(optimizer.state_dict(), save_path_optimizer)

                logs = {"loss_mlm": loss_mlm.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                with torch.no_grad():
                    mask_embeds_log=accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data[mask_token_ids].clone()
                    # mask_embeds_log=mask_embeds.clone()
                    masked_embeds_norm=torch.norm(mask_embeds_log,p=1).detach()
                    logs['masked_norm']=masked_embeds_norm.item()
                    # print(mask_embeds_copy.shape,'mask_embeds_copy.shape')
                    # print(mask_embeds.shape,'mask_embeds.shape')
                    logs['mask_sim']=cos_sim(mask_embeds_copy,mask_embeds_log).item()
                if args.report_to=='wandb' and accelerator.is_main_process:
                    wandb.log(logs)
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                progress_bar.update(1)
                global_step += 1 
                if global_step >= args.max_train_steps:
                    break
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()