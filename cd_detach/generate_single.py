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
    
    if args.seed is not None:
        set_seed(args.seed)
    if args.learned_embed_path1 is not None:
        exp_name=args.learned_embed_path1.split('/')[-3]
        step=args.learned_embed_path1.split('-')[-1]
    else:
        exp_name=args.resume_path.split('/')[-3]
        step=args.resume_path.split('-')[-1]
    exp_name+='_s{}'.format(step)
    exp_dir=os.path.join(args.output_dir,exp_name)
    sample_dir = os.path.join(exp_dir,'generated')
    merged_dir = os.path.join(exp_dir,'merged')
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(merged_dir, exist_ok=True)
    if accelerator.is_main_process:
        print(exp_dir,'exp_dir')
        codepath=os.path.join(exp_dir,'src')
        # if os.path.exists(codepath) and 'tmp' not in codepath:
        #     assert False
        caption_path = os.path.join(args.output_dir,exp_name,'captions.json')
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
    # placeholder_token1 = [args.placeholder_token1]
    # placeholder_token2 = [args.placeholder_token2]
    # tokenizer.add_tokens(mask_tokens)
    # tokenizer.add_tokens(placeholder_token1)
    # text_encoder.resize_token_embeddings(len(tokenizer))
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
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))
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
    
    # unet=unet.to(accelerator.device)
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
    if args.learned_embed_path1 is not None:
        pipeline.load_textual_inversion(args.learned_embed_path1,token=args.placeholder_token1, weight_name="{}.bin".format(args.placeholder_token1))
        print('learned_embed_path loaded')
    else:
        pipeline.load_textual_inversion(args.resume_path,token=args.placeholder_token1, weight_name="{}.bin".format(args.placeholder_token1))
    if args.resume_path and args.resume_path!='None':
        cd_layers_path=os.path.join(args.resume_path,'custom_diffusion.pt')
        saved_state_dict = torch.load(cd_layers_path, map_location=torch.device('cpu'))
        # for key in saved_state_dict:
        #     print(key,'saved')
        print()
        print()
        defined_state_dict=pipeline.unet.state_dict()
        new_state_dict={}
        for key in defined_state_dict:
            # print(key,'defined')
            if key in saved_state_dict:
                new_state_dict[key]=saved_state_dict[key]
            else:
                new_state_dict[key]=defined_state_dict[key]
        pipeline.unet.load_state_dict(new_state_dict,strict=True)
        print('unet parameters loaded')
        del new_state_dict
    if args.include_prior_concept:
        placeholder='{} {}'.format(args.placeholder_token1,args.prior_concept1)
    else:
        placeholder='{}'.format(args.placeholder_token1)
    if args.prompt_type=='nonliving':
        eval_prompts = [ 
            'a {0} in the jungle'.format(placeholder),
            'a {0} in the snow'.format(placeholder),
            'a {0} on the beach'.format(placeholder),
            'a {0} on a cobblestone street'.format(placeholder),
            'a {0} on top of pink fabric'.format(placeholder),
            'a {0} on top of a wooden floor'.format(placeholder),
            'a {0} with a city in the background'.format(placeholder),
            'a {0} with a mountain in the background'.format(placeholder),
            'a {0} with a blue house in the background'.format(placeholder),
            'a {0} on top of a purple rug in a forest'.format(placeholder),

            'a {0} with a wheat field in the background'.format(placeholder),
            'a {0} with a tree and autumn leaves in the background'.format(placeholder),
            'a {0} with the Eiffel Tower in the background'.format(placeholder),
            'a {0} floating on top of water'.format(placeholder),
            'a {0} floating in an ocean of milk'.format(placeholder),
            'a {0} on top of green grass with sunflowers around it'.format(placeholder),
            'a {0} on top of a mirror'.format(placeholder),
            'a {0} on top of the sidewalk in a crowded street'.format(placeholder),
            'a {0} on top of a dirt road'.format(placeholder),
            'a {0} on top of a white rug'.format(placeholder),

            'a red {0}'.format(placeholder),
            'a purple {0}'.format(placeholder),
            'a shiny {0}'.format(placeholder),
            'a wet {0}'.format(placeholder),
            'a cube shaped {0}'.format(placeholder)
            ]*args.num_images_per_prompt
    elif args.prompt_type=='pet':
        eval_prompts = [ 
        'a {0} in the jungle'.format(placeholder),
        'a {0} in the snow'.format(placeholder),
        'a {0} on the beach'.format(placeholder),
        'a {0} on a cobblestone street'.format(placeholder),
        'a {0} on top of pink fabric'.format(placeholder),
        'a {0} on top of a wooden floor'.format(placeholder),
        'a {0} with a city in the background'.format(placeholder),
        'a {0} with a mountain in the background'.format(placeholder),
        'a {0} with a blue house in the background'.format(placeholder),
        'a {0} on top of a purple rug in a forest'.format(placeholder),

        'a {0} wearing a red hat'.format(placeholder),
        'a {0} wearing a santa hat'.format(placeholder),
        'a {0} wearing a rainbow scarf'.format(placeholder),
        'a {0} wearing a black top hat and a monocle'.format(placeholder),
        'a {0} in a chef outfit'.format(placeholder),
        'a {0} in a firefighter outfit'.format(placeholder),
        'a {0} in a police outfit'.format(placeholder),
        'a {0} wearing pink glasses'.format(placeholder),
        'a {0} wearing a yellow shirt'.format(placeholder),
        'a {0} in a purple wizard outfit'.format(placeholder),
        'a red {0}'.format(placeholder),
        'a purple {0}'.format(placeholder),
        'a shiny {0}'.format(placeholder),
        'a wet {0}'.format(placeholder),
        'a cube shaped {0}'.format(placeholder)
        ]*args.num_images_per_prompt
    elif args.prompt_type=='building':
       eval_prompts=['photo of a {}'.format(placeholder),
       '{} in snowy ice'.format(placeholder),
       '{} in the fall season with leaves all around'.format(placeholder),
       '{} at a beach with a view of the seashore'.format(placeholder),
       'Photo of the {} with the sun rising in the sky.'.format(placeholder),
       '{} with forest in the background.'.format(placeholder),
       'puppy in front of the {}'.format(placeholder),
       'cat sitting in front of the {}'.format(placeholder),
       'cat sitting in front of {} in snowy ice'.format(placeholder),
       'squirrel in front of the {}'.format(placeholder),
       '{} oil painting ghibli inspired'.format(placeholder),
       '{} painting by artist claude monet'.format(placeholder),
       '{} digital painting 3d render geometric style'.format(placeholder),
       "Georgia O'Keeffe style {} painting".format(placeholder),
       'a watercolor painting of the {}'.format(placeholder),
       'painting of {} in the style of van gogh'.format(placeholder),
       'A futuristic {}'.format(placeholder),
       'A surreal landscape, {}'.format(placeholder),
       'A close up shot of the {}'.format(placeholder),
       'Top view of the {}'.format(placeholder)]*args.num_images_per_prompt
    else:
        assert False
    # batch_size=len(prompts)
    batch_size=args.eval_batch_size
    num_batches=(len(eval_prompts)//batch_size)+int((len(eval_prompts)/batch_size)>0)
    count=0
    validation_files=sorted(os.listdir(args.train_data_dir1))
    validation_target=Image.open(os.path.join((args.train_data_dir1),validation_files[0])).resize((512,512)).convert('RGB')
    caption_data={}
    with torch.no_grad():
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
                prompt_saved=prompt.replace(placeholder,args.prior_concept1)
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
            
    json.dump(caption_data,caption_file,indent=1)
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
    exit()
