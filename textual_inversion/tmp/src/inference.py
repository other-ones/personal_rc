

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


def save_progress(text_encoder, placeholder_token_id, accelerator, args, save_path):
    logger.info("Saving embeddings")
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
            .get_input_embeddings()
            .weight[placeholder_token_id]
    )
    print("Current Learned Embeddings: ", learned_embeds[:4])
    print("saved to ", save_path)
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}
    torch.save(learned_embeds_dict, save_path)

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
    exp_name=args.learned_embed_path.split('/')[1]
    file_name=args.learned_embed_path.split('/')[-1].split('.')[0]
    stepname=file_name.split('_')[-1]
    sample_dir = os.path.join(args.output_dir,'step_'+stepname)
    os.makedirs(sample_dir, exist_ok=True)
    meta_path=os.path.join(args.output_dir,'meta.txt')
    
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
    placeholder_token2 = [args.placeholder_token1]
    tokenizer.add_tokens(mask_tokens)
    tokenizer.add_tokens(placeholder_token1)
    tokenizer.add_tokens(placeholder_token2)
    placeholder_token_id1 = tokenizer.convert_tokens_to_ids(placeholder_token1)
    placeholder_token_id2 = tokenizer.convert_tokens_to_ids(placeholder_token2)
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    print(token_embeds.shape,'token_embeds.shape')
    learned_embed1=torch.load(args.learned_embed_path1)#[args.placeholder_token]
    learned_embed1=learned_embed1[args.placeholder_token1]
    learned_embed2=torch.load(args.learned_embed_path2)#[args.placeholder_token]
    learned_embed2=learned_embed2[args.placeholder_token2]
    with torch.no_grad():
        token_embeds[placeholder_token_id1] = learned_embed1 #token_embeds[initializer_token_id].clone()
        token_embeds[placeholder_token_id2] = learned_embed2 #token_embeds[initializer_token_id].clone()
    # if args.mask_embed_path is not None:
    #     mask_embeds=torch.load(args.mask_embed_path)[args.mask_tokens].to(accelerator.device)
    #     mask_embeds=F.normalize(mask_embeds,p=1,dim=-1)*args.avg_norm
    # Freeze all parameters except for the token embeddings in text encoder
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


    if accelerator.is_main_process:
        print('unet param loaded')
    (unet
    #  test_dataloader
     ) = accelerator.prepare(
                    unet
                    # test_dataloader
                    )
    std_pipeline = StableDiffusionPipeline.from_pretrained( model_name,
                            unet=accelerator.unwrap_model(unet, **extra_args),
                            tokenizer=accelerator.unwrap_model(tokenizer, **extra_args),
                            )
    std_scheduler = std_pipeline.scheduler
    std_fe_extractor = std_pipeline.feature_extractor
    del std_pipeline
    # unet.eval()
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
            scheduler=accelerator.unwrap_model(std_scheduler, **extra_args),
            feature_extractor=accelerator.unwrap_model(std_fe_extractor, **extra_args),
            safety_checker=None,
            requires_safety_checker=False,
        )
    # prompts=[
    #     "a picture of {} with the Eiffel Tower in the background".format(args.placeholder_token),
    #     "a picture of {} with the Great Wall of China in the background".format(args.placeholder_token),
    #     "a picture of {} with a deep ocean trench in the background".format(args.placeholder_token),
    #     "a picture of {} with a moonlit valley in the background".format(args.placeholder_token),
    #     "a picture of {} with a sistine chapel in the background".format(args.placeholder_token),
    #     ]
    prompts=[
        "a picture of {} {} with the Great Wall of China in the background".format(args.placeholder_token,args.prior_concept),
        ]*7
    print(prompts[0],'prompts')
    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    batch_size=len(prompts)
    validation_files=os.listdir(args.train_data_dir)
    validation_target=Image.open(os.path.join((args.train_data_dir),validation_files[0])).resize((512,512))
    with torch.no_grad():
        images = pipeline(prompt=prompts, 
                          num_inference_steps=50, 
                          guidance_scale=7.5, width=512, height=512).images

    merged_viz = Image.new('RGB', (512*(batch_size+1), 512), (255, 255, 255))
    merged_viz.paste(validation_target.convert('RGB'),(0,0))
    for iidx in range(batch_size):
        image=images[iidx]
        merged_viz.paste(image.convert('RGB'),((iidx+1)*512,0))

    rand_num=np.random.randint(0,10000)
    img_path=os.path.join(sample_dir,'{}.jpg'.format(timestr))
    print(sample_dir,'sample_dir')
    print(img_path,'img_path')

    merged_viz.save(img_path)   
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
    exit()
