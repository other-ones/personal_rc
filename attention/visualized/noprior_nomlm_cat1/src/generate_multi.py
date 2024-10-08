


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

from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, AutoProcessor

from lora_diffusion.xformers_utils import set_use_memory_efficient_attention_xformers
from torch.utils.data import Dataset, SubsetRandomSampler
from torchvision import transforms
from pathlib import Path
import random
from PIL import Image
from lora_diffusion import tune_lora_scale, patch_pipe
import torchvision.transforms as T
import inspect
from azureml.core import Run
run = Run.get_context()
import socket
hostname = socket.gethostname()

from configs import parse_args
from torch import nn
# added
from utils import render_caption
import time
# added


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
    if args.learned_embed_path_multi:
        exp_name=args.learned_embed_path_multi.split('/')[3]
    elif (args.learned_embed_path1 and args.learned_embed_path2 and args.exp_name):
        exp_name=args.exp_name
    else:
        assert False,'provide name of the destination'
    exp_dir=os.path.join(args.output_dir,exp_name)
    sample_dir = os.path.join(exp_dir,'generated')
    merged_dir = os.path.join(exp_dir,'merged')
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(merged_dir, exist_ok=True)
    if accelerator.is_main_process:
        print(exp_dir,'exp_dir')
        codepath=os.path.join(exp_dir,'src')
        if os.path.exists(codepath) and 'tmp' not in codepath:#and 'tmp' not in codepath:
            assert False
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
    mask_tokens = [args.mask_tokens]
    placeholder_token1 = [args.placeholder_token1]
    placeholder_token2 = [args.placeholder_token2]
    # placeholder_token2 = [args.placeholder_token2]
    tokenizer.add_tokens(mask_tokens)
    tokenizer.add_tokens(placeholder_token1)
    tokenizer.add_tokens(placeholder_token2)
    placeholder_token_id1 = tokenizer.convert_tokens_to_ids(placeholder_token1)
    placeholder_token_id2 = tokenizer.convert_tokens_to_ids(placeholder_token2)
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    print(token_embeds.shape,'token_embeds.shape')

    if args.learned_embed_path_multi:
        dict_multi=torch.load(args.learned_embed_path_multi)#[args.placeholder_token1]
        learned_embed1=dict_multi[args.placeholder_token1]
        learned_embed2=dict_multi[args.placeholder_token2]
    elif (args.learned_embed_path1 and args.learned_embed_path2):
        dict1=torch.load(args.learned_embed_path1)
        dict2=torch.load(args.learned_embed_path2)
        learned_embed1=dict1[args.placeholder_token1]
        learned_embed2=dict2[args.placeholder_token2]
    else:
        assert False,'learned_embed path unprovided'
    with torch.no_grad():
        token_embeds[placeholder_token_id1] = learned_embed1 #
        token_embeds[placeholder_token_id2] = learned_embed2 #
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
    # HERE
    
    """VAE Initialization"""
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )
    """UNet Initialization"""
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

    if accelerator.is_main_process:
        print('unet param loaded')
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
    # if args.augmenter_path1 and args.augmenter_path1!='None':
    #     state_dict = torch.load(args.augmenter_path1, map_location=torch.device('cpu'))
    #     augmenter.load_state_dict(state_dict,strict=True)
    #     print('augmenter parameters loaded')
    #     del state_dict
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
        placeholder1='{} {}'.format(args.placeholder_token1,args.prior_concept1)
        placeholder2='{} {}'.format(args.placeholder_token2,args.prior_concept2)
    else:
        placeholder1='{}'.format(args.placeholder_token1)
        placeholder2='{}'.format(args.placeholder_token2)
    if args.prompt_type=='two_pets':
        eval_prompts = [
            'A photo of the {0} and {1}'.format(placeholder1,placeholder2),
            'the {0} and {1} sitting on an antique table'.format(placeholder1,placeholder2),
            'the {0} and {1} sitting on beach with a view of seashore'.format(placeholder1,placeholder2),
            '{1} and {0} side by side on a mountaintop, overlooking a sunrise'.format(placeholder1,placeholder2),
            'photo of {0} and {1} on a surfboard together in the middle of a clear blue ocean'.format(placeholder1,placeholder2),
            '{0} playing with {1}'.format(placeholder1,placeholder2),
            'A pointillist painting of the {0} and {1}'.format(placeholder1,placeholder2),
            'An impressionist painting of the {0} and {1} sitting on a sofa'.format(placeholder1,placeholder2),
            'A black-and-white photograph of the {1} playing with {0}'.format(placeholder1,placeholder2),
            '{0} playing with a robot toy {1}'.format(placeholder1,placeholder2),
            '{1} playing with a robot toy {0}'.format(placeholder1,placeholder2),
            'A plush toy replica of the {0} and {1} sitting beside it'.format(placeholder1,placeholder2),
            ]*args.num_images_per_prompt
    else:
        assert False
    # batch_size=len(prompts)
    batch_size=args.eval_batch_size
    num_batches=(len(eval_prompts)//batch_size)+int((len(eval_prompts)/batch_size)>0)
    count=0
    validation_files1=sorted(os.listdir(args.train_data_dir1))
    validation_files2=sorted(os.listdir(args.train_data_dir2))
    validation_target1=Image.open(os.path.join((args.train_data_dir1),validation_files1[0])).resize((512,512)).convert('RGB')
    validation_target2=Image.open(os.path.join((args.train_data_dir2),validation_files2[0])).resize((512,512)).convert('RGB')
    caption_data={}
    learned_embed1=learned_embed1.to(accelerator.device)
    learned_embed2=learned_embed2.to(accelerator.device)
    with torch.no_grad():

        if args.normalize_target1:
            target_emb1=F.normalize(learned_embed1,p=1,dim=-1)*args.normalize_target1
        else:
            target_emb1=learned_embed1
        if args.normalize_target2:
            target_emb2=F.normalize(learned_embed2,p=1,dim=-1)*args.normalize_target2
        else:
            target_emb2=learned_embed2
        for batch_idx in range(num_batches):
            prompts=eval_prompts[batch_idx*batch_size:(batch_idx+1)*batch_size]
            if not len(prompts):
                break
            is_keyword_tokens_list1=[]
            is_keyword_tokens_list2=[]
            for prompt in prompts:
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
                # print(is_keyword_tokens1.shape,'is_keyword_tokens1.shape')
                # print(is_keyword_tokens2.shape,'is_keyword_tokens2.shape')
                is_keyword_tokens_list1.append(is_keyword_tokens1)
                is_keyword_tokens_list2.append(is_keyword_tokens2)

            is_keyword_tokens_list1=torch.stack(is_keyword_tokens_list1)
            is_keyword_tokens_list2=torch.stack(is_keyword_tokens_list2)
            # print(sample_dir,'sample_dir')
            # print(is_keyword_tokens_list1.shape,'is_keyword_tokens_list1.shape')
            # print(is_keyword_tokens_list2.shape,'is_keyword_tokens_list2.shape')
            images = pipeline(prompt=prompts, 
                            num_inference_steps=50, 
                            guidance_scale=7.5, width=512, height=512,
                            num_images_per_prompt=1,
                            is_keyword_tokens1=is_keyword_tokens_list1,
                            is_keyword_tokens2=is_keyword_tokens_list2,
                            inj_embeddings1=target_emb1,
                            inj_embeddings2=target_emb2,
                            ).images
            num_cols=5
            num_viz_samples=5
            num_rows=len(images[:num_viz_samples])//num_cols
            num_rows=max(1,num_rows)
            margin_bottom=150
            margin_right=10
            merged_viz = Image.new('RGB', ((512+margin_right)*(num_cols+2), (512+margin_bottom)*num_rows), (255, 255, 255))
            for ridx in range(num_rows):
                merged_viz.paste(validation_target1,(0,ridx*(512+margin_bottom)))
                merged_viz.paste(validation_target2,((512+margin_right),ridx*(512+margin_bottom)))    
            render_delay=0
            for iidx,(image, prompt) in enumerate(zip(images[:],prompts[:])):
                image_name='{:04d}'.format(count+1)
                img_path=os.path.join(sample_dir,'{}.jpg'.format(image_name))
                prompt_saved=prompt.replace(placeholder1,args.prior_concept1)
                prompt_saved=prompt_saved.replace(placeholder2,args.prior_concept2)
                caption_data[image_name]=prompt_saved
                st=time.time()
                render_delay+=(time.time()-st)
                image.save(img_path)
                count+=1
            for iidx,(image, prompt) in enumerate(zip(images[:num_viz_samples],prompts[:num_viz_samples])):
                row_idx=iidx//num_cols
                col_idx=iidx-(num_cols*row_idx)
                x0=(col_idx+2)*(512+margin_right)
                y0=row_idx*(512+margin_bottom)+512
                x1=x0+(512+margin_right)
                y1=y0+margin_bottom
                st=time.time()
                merged_viz=render_caption(merged_viz,prompt,[x0,y0+20,x1,y1])
                render_delay+=(time.time()-st)
                merged_viz.paste(image.convert('RGB'),((col_idx+2)*(512+margin_right),row_idx*(512+margin_bottom)))
                count+=1
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
