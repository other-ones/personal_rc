from datetime import datetime
from utils import render_caption
import time


import torchvision.ops.roi_align as roi_align
import json
from torch.utils.data.distributed import DistributedSampler as DistributedSampler
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import re
from collections import OrderedDict
# Bootstrapped from:
# from utils import random_crop_image_mask,create_random_mask,create_mask_from_coords
# from transformers import CLIPVisionModel, CLIPVisionConfig, CLIPProcessor
import numpy as np
import sys
sys.path.insert(0, './packages')
import argparse
import hashlib
import itertools
import math
import os
import inspect
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import pdb
from accelerate import Accelerator
# from accelerate import PartialState
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    StableDiffusionPipeline,
    UNet2DModel
)
from diffusers.models.attention_processor import (
    CustomDiffusionAttnProcessor,
    CustomDiffusionAttnProcessor2_0,
    CustomDiffusionXFormersAttnProcessor,
)
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, AutoProcessor

from torch.utils.data import Dataset, SubsetRandomSampler
from torchvision import transforms
from pathlib import Path
import random
from PIL import Image
import torchvision.transforms as T
import inspect
import socket
hostname = socket.gethostname()

from configs import parse_args
from torch import nn
# ADDED
torch.use_deterministic_algorithms(True)
# ADDED

def to_img(x, clip=True):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 32, x.size(3))
    if clip:
        x = torch.clip(x, 0, 1)
    return x


def _randomset(lis):
    ret = []
    for i in range(len(lis)):
        if random.random() < 0.5:
            ret.append(lis[i])
    return ret


def _shuffle(lis):
    return random.sample(lis, len(lis))

image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




logger = get_logger(__name__)



def freeze_params(params):
    for param in params:
        param.requires_grad = False


def unfreeze_params(params):
    for param in params:
        param.requires_grad = True


def main(args):
    model_name = args.pretrained_model_name_or_path#'stabilityai/stable-diffusion-2-1'
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    np.random.seed(int(time.time())) 
    # if args.seed is not None:
    #     set_seed(args.seed)
    if args.learned_embed_path1 is not None:
        exp_name=args.learned_embed_path1.split('/')[-3]
        step=args.learned_embed_path1.split('-')[-1]
    else:
        exp_name=args.resume_cd_path.split('/')[-3]
        step=args.resume_cd_path.split('-')[-1]
    exp_name+='_s{}'.format(step)
    # exp_dir=os.path.join(args.output_dir,exp_name)
    exp_dir=args.dst_exp_path

    # exp_dir+='_{:06d}'.format(np.random.randint(low=0,high=99999))
    current_time = datetime.now()
    time_str = current_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    exp_dir+=f'_{time_str}'
    sample_dir = os.path.join(exp_dir,'generated')
    merged_dir = os.path.join(exp_dir,'merged')
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(merged_dir, exist_ok=True)
    if accelerator.is_main_process:
        print(exp_dir,'exp_dir')
        codepath=os.path.join(exp_dir,'src')
        if os.path.exists(codepath) and 'tmp' not in codepath:
            assert False
        caption_path = os.path.join(exp_dir,'captions.json')
        caption_file=open(caption_path,'w')
        os.makedirs(codepath,exist_ok=True)
        os.system('cp *.py {}'.format(codepath))
        os.system('cp datasets_pkgs {} -R'.format(codepath))
        os.system('cp packages {} -R'.format(codepath))
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

    
    
    # """original text_encoder"""
    
    tokenizer = CLIPTokenizer.from_pretrained(
        model_name,
        subfolder="tokenizer",
        revision=args.revision,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_name,
        subfolder="text_encoder",
        revision=args.revision,
    )
    # HERE
    # mask_tokens = [args.mask_tokens]
    # tokenizer.add_tokens(mask_tokens)
    placeholder_token1 = [args.placeholder_token1]
    tokenizer.add_tokens(placeholder_token1)
    placeholder_token_id1 = tokenizer.convert_tokens_to_ids(placeholder_token1)
    if 'nomlm' not in  args.resume_cd_path:
        mask_tokens = [args.mask_tokens]
        tokenizer.add_tokens(mask_tokens)
        mask_token_ids = tokenizer.convert_tokens_to_ids(mask_tokens)
    else:
        mask_token_ids=None
    text_encoder.resize_token_embeddings(len(tokenizer))
    text_encoder.requires_grad_(False)
    # text_encoder.text_model.encoder.requires_grad_(False)
    # text_encoder.text_model.final_layer_norm.requires_grad_(False)
    # text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
    # text_encoder.text_model.encoder.requires_grad_(False)
    # text_encoder.text_model.final_layer_norm.requires_grad_(False)
    # text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
    # HERE
    
    """VAE Initialization"""
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )
    """UNet Initialization"""

    # Custom Diffusion Layers
    st = unet.state_dict()
    train_q_out = False if args.freeze_model == "crossattn_kv" else True
    custom_diffusion_attn_procs = {}
    attention_class = (
        CustomDiffusionAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else CustomDiffusionAttnProcessor
    )
    train_kv = True
    for name, _ in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        layer_name = name.split(".processor")[0]
        weights = {
            "to_k_custom_diffusion.weight": st[layer_name + ".to_k.weight"],
            "to_v_custom_diffusion.weight": st[layer_name + ".to_v.weight"],
        }
        if train_q_out:
            weights["to_q_custom_diffusion.weight"] = st[layer_name + ".to_q.weight"]
            weights["to_out_custom_diffusion.0.weight"] = st[layer_name + ".to_out.0.weight"]
            weights["to_out_custom_diffusion.0.bias"] = st[layer_name + ".to_out.0.bias"]
        if cross_attention_dim is not None:
            custom_diffusion_attn_procs[name] = attention_class(
                train_kv=train_kv,
                train_q_out=train_q_out,
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
            ).to(accelerator.device)
            custom_diffusion_attn_procs[name].load_state_dict(weights)
        else:
            custom_diffusion_attn_procs[name] = attention_class(
                train_kv=False,
                train_q_out=False,
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
            )
    unet.set_attn_processor(custom_diffusion_attn_procs)
    # custom_diffusion_layers = AttnProcsLayers(pipe.unet.attn_processors)
    del st

    # Custom Diffusion Layers


    





    print(inspect.getsourcefile(UNet2DConditionModel.from_pretrained), 'inspect')
    for param in unet.parameters():
        param.requires_grad = False


    vae.requires_grad_(False)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    # if accelerator.is_main_process:
    #     accelerator.init_trackers("dreambooth", config=vars(args))
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

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    (unet,
     noise_scheduler,
     text_encoder,
     vae,
     ) = accelerator.prepare(
                    unet,
                    noise_scheduler,
                    text_encoder,
                    vae,
                    )
    
    
    
    
    if args.learned_embed_path1 is not None:
        token_embeds = text_encoder.get_input_embeddings().weight.data
        learned_embed1=torch.load(args.learned_embed_path1)#[args.placeholder_token]
        print('load learned embeddings')
        learned_embed1=learned_embed1[args.placeholder_token1]
        learned_embed1=learned_embed1.to(accelerator.device)
        # initial_embed=learned_embed1.clone().detach()
        with torch.no_grad():
            token_embeds[placeholder_token_id1] = learned_embed1.clone()
        del learned_embed1
    if args.resume_cd_path and args.resume_cd_path!='None':
        saved_state_dict = torch.load(args.resume_cd_path, map_location=torch.device('cpu'))
        print()
        print()
        defined_state_dict=unet.state_dict()
        new_state_dict={}
        for key in defined_state_dict:
            # print(key,'defined')
            if key in saved_state_dict:
                new_state_dict[key]=saved_state_dict[key]
            else:
                new_state_dict[key]=defined_state_dict[key]
        unet.load_state_dict(new_state_dict,strict=True)
        print('unet parameters loaded')
        del new_state_dict
    
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
    unet=unet.to(accelerator.device)
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
        
        
    if args.include_prior_concept:
        placeholder='{} {}'.format(args.placeholder_token1,args.train_prior_concept1)
    else:
        placeholder='{}'.format(args.placeholder_token1)
    # eval_prompts=json.load(open(args.benchmark_path))[args.eval_prompt_type]
    eval_prompts=[args.teaser_prompt.format(placeholder)]
    eval_prompts=eval_prompts*args.num_images_per_prompt
    batch_size=args.eval_batch_size
    num_batches=(len(eval_prompts)//batch_size)+int((len(eval_prompts)/batch_size)>0)
    count=0
    validation_files=sorted(os.listdir(args.train_data_dir1))
    validation_target=Image.open(os.path.join((args.train_data_dir1),validation_files[0])).resize((512,512)).convert('RGB')
    caption_data={}
    with torch.no_grad():
        target_emb1=accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[min(placeholder_token_id1) : max(placeholder_token_id1) + 1]
        for batch_idx in range(num_batches):
            prompts=eval_prompts[batch_idx*batch_size:(batch_idx+1)*batch_size]
            if not len(prompts):
                break

            
            print(sample_dir,'sample_dir')
            images = pipeline(prompt=prompts, 
                            num_inference_steps=50, 
                            guidance_scale=7.5, width=512, height=512,
                            num_images_per_prompt=1,
                            ).images
            
            # 
            num_cols=5
            num_viz_samples=5
            num_rows=len(images[:num_viz_samples])//num_cols
            num_rows=max(1,num_rows)
            margin_bottom=150
            margin_right=10
            render_delay=0
            merged_viz = Image.new('RGB', ((512+margin_right)*(num_cols+1), (512+margin_bottom)*num_rows), (255, 255, 255))
            for ridx in range(num_rows):
                merged_viz.paste(validation_target,(0,ridx*(512+margin_bottom)))
            for iidx,(image, prompt) in enumerate(zip(images[:],prompts[:])):
                image_name='{:04d}'.format(count+1)
                img_path=os.path.join(sample_dir,'{}.jpg'.format(image_name))
                prompt_saved=prompt.replace(placeholder,args.eval_prior_concept1)
                caption_data[image_name]=prompt_saved
                st=time.time()
                render_delay+=(time.time()-st)
                image.save(img_path)
                count+=1
            for iidx,(image, prompt) in enumerate(zip(images[:num_viz_samples],prompts[:num_viz_samples])):
                row_idx=iidx//num_cols
                col_idx=iidx-(num_cols*row_idx)
                x0=(col_idx+1)*(512+margin_right)
                y0=row_idx*(512+margin_bottom)+512
                x1=x0+(512+margin_right)
                y1=y0+margin_bottom
                st=time.time()
                merged_viz=render_caption(merged_viz,prompt,[x0,y0+20,x1,y1])
                render_delay+=(time.time()-st)
                merged_viz.paste(image.convert('RGB'),((col_idx+1)*(512+margin_right),row_idx*(512+margin_bottom)))
            print(batch_idx+1,num_batches,render_delay)
            print(merged_viz.size,'merged_viz.size',len(images),'len(images)')
            merged_viz.save(os.path.join(merged_dir,'merged_{:03d}.jpg'.format(batch_idx+1)))
            torch.cuda.empty_cache()
            
    json.dump(caption_data,caption_file,indent=1)
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
    exit()
