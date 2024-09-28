import cv2
import ptp_utils
import abc
from typing import Optional, Union, Tuple, List, Callable, Dict
import shutil
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
from pipeline_syngen import StableDiffusionPipelineSyngenSD
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

# from lora_diffusion.xformers_utils import set_use_memory_efficient_attention_xformers
from torch.utils.data import Dataset, SubsetRandomSampler
from torchvision import transforms
from pathlib import Path
import random
from PIL import Image
# from lora_diffusion import tune_lora_scale, patch_pipe
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

class LocalBlend:
    def __call__(self, x_t, attention_store):
        k = 1
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)
        mask = (mask[:1] + mask[1:]).float()
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], threshold=.3):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold


class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t
    def between_steps(self):
        return
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class EmptyControl(AttentionControl):
    
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn
class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

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

    exp_dir=args.dst_exp_path
    exp_dir+='_seed{}'.format(args.seed)
    load_fname=args.learned_embed_path1.split('/')[-1].split('.')[0]
    # sname=load_fname.split('_s')[-1]
    # exp_dir+='_s{}'.format(sname)
    sample_dir = os.path.join(exp_dir,'generated')
    merged_dir = os.path.join(exp_dir,'merged')
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(merged_dir, exist_ok=True)
    ce_log_file=open(os.path.join(exp_dir,'cross_entropy.json'),'w')
    if accelerator.is_main_process:
        print(exp_dir,'exp_dir')
        codepath=os.path.join(exp_dir,'src')
        # if os.path.exists(codepath) and 'tmp' not in codepath:
        #     assert False

        caption_path = os.path.join(exp_dir,'captions.json')
        caption_file=open(caption_path,'w')
        os.makedirs(codepath,exist_ok=True)
        os.system('cp *.py {}'.format(codepath))
        # os.system('cp datasets_pkgs {} -R'.format(codepath))
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

    
    
    # """original text_encoder"""
    
    tokenizer = CLIPTokenizer.from_pretrained(
        model_name,
        subfolder="tokenizer",
        revision=None,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_name,
        subfolder="text_encoder",
        revision=None,
    )

    # HERE
    token_embeds = text_encoder.get_input_embeddings().weight.data
    print(token_embeds.shape,'token_embeds.shape before')
    mask_tokens = [args.mask_tokens]
    placeholder_token1 = [args.placeholder_token1]
    # placeholder_token2 = [args.placeholder_token2]
    tokenizer.add_tokens(mask_tokens)
    tokenizer.add_tokens(placeholder_token1)
    placeholder_token_id1 = tokenizer.convert_tokens_to_ids(placeholder_token1)
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    print(token_embeds.shape,'token_embeds.shape after')
    learned_embed1=torch.load(args.learned_embed_path1)#[args.placeholder_token1]
    learned_embed1=learned_embed1[args.placeholder_token1]
    initializer_token_ids = tokenizer.encode(args.train_prior_concept1, add_special_tokens=False)
    initializer_token_id = initializer_token_ids[0]
    prior_embed=token_embeds[initializer_token_id].detach().clone()
    with torch.no_grad():
        token_embeds[placeholder_token_id1] = learned_embed1 #
        # token_embeds[placeholder_token_id2] = learned_embed2 #token_embeds[initializer_token_id].clone()
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
    # HERE
    
    """VAE Initialization"""
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=None, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=None, variant=args.variant
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
    # HERE
    syngen_pipeline = StableDiffusionPipelineSyngenSD.from_pretrained( model_name,
                            unet=accelerator.unwrap_model(unet, **extra_args),
                            tokenizer=accelerator.unwrap_model(tokenizer, **extra_args),
                            text_encoder=accelerator.unwrap_model(text_encoder, **extra_args),
                            vae=accelerator.unwrap_model(vae, **extra_args),
                            )
    if args.include_prior_concept:
        placeholder="{} {}".format(args.placeholder_token1,args.train_prior_concept1)
    else:
        placeholder="{}".format(args.placeholder_token1)
    eval_prompts=json.load(open(args.benchmark_path))[args.eval_prompt_type]
    eval_prompts=eval_prompts[:1]
    eval_prompts_aug=[item.format(placeholder) for item in eval_prompts]
    eval_prompts_raw=[item.format(placeholder).replace('<k>','') for item in eval_prompts]
    eval_prompts_save=[item.format(args.eval_prior_concept1).replace('<k>','') for item in eval_prompts]

    eval_prompts_aug=eval_prompts_aug*args.num_images_per_prompt
    eval_prompts_raw=eval_prompts_raw*args.num_images_per_prompt
    batch_size=args.eval_batch_size
    num_batches=(len(eval_prompts)//batch_size)+int((len(eval_prompts)/batch_size)>0)
    ce_data={}
    caption_data={}
    count=0

    for batch_idx in range(num_batches):
        prompts_aug=eval_prompts_aug[batch_idx*batch_size:(batch_idx+1)*batch_size]
        prompts_raw=eval_prompts_raw[batch_idx*batch_size:(batch_idx+1)*batch_size]
        prompts_save=eval_prompts_save[batch_idx*batch_size:(batch_idx+1)*batch_size]

        is_keyword_tokens_list=[]
        is_bg_tokens_list=[]
        non_special_idxs_list=[]

        
        
        for gen_idx in range(len(prompts_aug)):
            prompt_aug=prompts_aug[gen_idx]
            is_keyword_tokens=[False]
            is_bg_tokens=[False]
            non_special_idxs=[False]
            words_aug=prompt_aug.split()
            for word_idx in range(len(words_aug)):
                cap_word_aug=words_aug[word_idx]
                cap_word_raw=cap_word_aug.replace('<k>','')
                word_token_ids=tokenizer.encode(cap_word_raw,add_special_tokens=False)
                num_tokens=len(word_token_ids)
                non_special_idxs+=([True]*num_tokens)
                for tok_id in word_token_ids:
                    if '<k>' in cap_word_aug:
                        is_bg_tokens.append(True)
                    else:
                        is_bg_tokens.append(False)
                    tok_decoded=tokenizer.decode(tok_id)
                    if tok_decoded==args.placeholder_token1:
                        is_keyword_tokens.append(True)
                    else:
                        is_keyword_tokens.append(False)
            non_special_idxs=non_special_idxs[:tokenizer.model_max_length-1]
            for _ in range(len(non_special_idxs),tokenizer.model_max_length):
                non_special_idxs.append(False)   
            for _ in range(len(is_keyword_tokens),tokenizer.model_max_length):
                is_keyword_tokens.append(False)
            for _ in range(len(is_bg_tokens),tokenizer.model_max_length):
                is_bg_tokens.append(False)
            assert len(is_keyword_tokens)==tokenizer.model_max_length
            if sum(is_keyword_tokens)!=1:
                print(sum(is_keyword_tokens),'is_keyword_tokens')
            assert sum(is_keyword_tokens)==1,'sum(keywords)==1'

            non_special_idxs=torch.BoolTensor(non_special_idxs)
            is_keyword_tokens=torch.BoolTensor(is_keyword_tokens)
            is_bg_tokens=torch.BoolTensor(is_bg_tokens)
            non_special_idxs_list.append(non_special_idxs)
            is_keyword_tokens_list.append(is_keyword_tokens)
            is_bg_tokens_list.append(is_bg_tokens)
        non_special_idxs_list=torch.stack(non_special_idxs_list)
        is_keyword_tokens_list=torch.stack(is_keyword_tokens_list)
        is_bg_tokens_list=torch.stack(is_bg_tokens_list)


        attn_res=256
        ce_criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        image_list,attention_maps_list=syngen_pipeline(
        # image_list=syngen_pipeline(
                            prompts_raw,
                            height=512,width=512,
                            num_inference_steps=50,
                            attn_res=(int(math.sqrt(attn_res)), int(math.sqrt(attn_res))),
                            attn_mod_params=None,
                            guidance_scale=7.5,
                            verbose=True,
                            )
        # exit(0)
        image_list=image_list.images
        print(len(attention_maps_list),'len(attention_maps_list)') # eval_batch_size
        for img_idx,img in enumerate(image_list):
            is_keywords_tokens=is_keyword_tokens_list[img_idx]
            is_bg_tokens=is_bg_tokens_list[img_idx]
            non_special_idxs=non_special_idxs_list[img_idx]
            prompt_raw=prompts_raw[img_idx]
            prompt_save=prompts_save[img_idx]
            prompt=prompts_raw[img_idx]
            count+=1
            img_name='{:03d}'.format(count)
            caption_data[img_name]=prompt_raw
            img.save(os.path.join(sample_dir,'{}.png'.format(img_name)))
            attention_maps=attention_maps_list[img_idx] # attn maps per image (77)
            input_ids = syngen_pipeline.tokenizer(prompt_raw).input_ids
            input_ids=tokenizer.pad(
                        {"input_ids": input_ids},
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        return_tensors="pt",
                    ).input_ids
            tokens = {i: tok 
                        for tok, i in zip(syngen_pipeline.tokenizer.convert_ids_to_tokens(input_ids), range(len(input_ids)))
                    }
            # print(is_keywords_tokens.shape,'is_keywords_tokens.shape')
            # print(input_ids.shape,'input_ids.shape')
            if img_idx==0:
                # Log Decoding
                input_ids_key=input_ids[is_keywords_tokens]
                input_ids_bg=input_ids[is_bg_tokens]
                print(input_ids_key.shape,'input_ids_key.shape')
                print(input_ids_bg.shape,'input_ids_bg.shape')
                decoded_key=tokenizer.batch_decode(input_ids_key)
                decoded_bg=tokenizer.batch_decode(input_ids_bg)
                decoded_key_list=[]
                decoded_bg_list=[]
                num_logs=10
                print()
                print('batch{}'.format(batch_idx+1))
                print()
                dots='-'*100
                print(dots)
                print(dots)
                for dec1 in decoded_key:
                    decoded_key_list.append('{:8}'.format(dec1))
                for dec_bg in decoded_bg:
                    decoded_bg_list.append('{:8}'.format(dec_bg))

                decoded_key=' '.join(decoded_key_list[:num_logs])
                decoded_bg=' '.join(decoded_bg_list[:num_logs])

                decoded=tokenizer.batch_decode(input_ids[non_special_idxs])
                decoded_list=[]
                for dec in decoded:
                    decoded_list.append('{:8}'.format(dec))
                decoded=' '.join(decoded_list)
                print('Input\t\t|{}'.format(decoded))
                print(dots)
                print('Key1\t\t|{}'.format(decoded_key))
                print('BG\t\t|{}'.format(decoded_bg))
                print(dots)
                print(dots)
                print()
                # Log Decoding


            attn_dir=os.path.join(exp_dir,'attns')
            prompt_splits=prompt_raw.split()
            prompt_cat='_'.join(prompt_splits)
            prompt_cat='{:03d}_'.format(count)+prompt_cat
            dst_attn_dir=os.path.join(attn_dir,prompt_cat)
            os.makedirs(dst_attn_dir,exist_ok=True)

            avg_entropy=[]
            # cross entropy 
            # print(len(attention_maps),'len(attntion_maps)') #77
            attention_maps=torch.stack(attention_maps)
            keyword_maps=attention_maps[is_keyword_tokens]
            bg_maps=attention_maps[is_bg_tokens]
            keyword_maps=keyword_maps.repeat(len(bg_maps),1,1)
            ce_loss=ce_criterion(keyword_maps,bg_maps).mean().item()
            ce_data[img_name]={"cross_entropy":ce_loss,"prompt":prompt_save}
            # cross entropy 
            # save image - attention map
            for kidx in range(len(is_keywords_tokens)):
                ktok=tokens[kidx].replace('</w>','')
                if ktok=="<|endoftext|>":
                    break
                attn_map=attention_maps[kidx].detach().cpu().numpy()#*args.vis_strength
                # print(attn_map.shape,'attn_map.shape') #16,16
                attn_map_flat = attn_map.reshape(-1)
                entropy = -np.sum(attn_map_flat * np.log(attn_map_flat + 1e-7))
                avg_entropy.append(entropy)
                attn_map=(255 * (attn_map / np.max(attn_map)))
                attn_map=cv2.resize(attn_map,(512,512))
                cv2.imwrite(os.path.join(dst_attn_dir,'{:03d}_{:02d}_{}.png'.format(count,kidx,ktok)),attn_map)
            # save image - attention map
            torch.cuda.empty_cache()
            avg_entropy=np.mean(avg_entropy)
            print(prompt,avg_entropy,'avg_entropy')
            print(exp_dir)
            break
    # for item in attention_maps_list:
    #     print(item.shape,'attention_maps_list.shape')
    # print(attention_maps_list,'attention_maps_list')
    json.dump(ce_data,ce_log_file,indent=1)
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
    exit()
