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
    
    if args.seed is not None:
        # set_seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        random.seed(args.seed)

    exp_dir=args.dst_exp_path
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
    # placeholder_token2 = [args.placeholder_token2]
    tokenizer.add_tokens(mask_tokens)
    tokenizer.add_tokens(placeholder_token1)
    placeholder_token_id1 = tokenizer.convert_tokens_to_ids(placeholder_token1)
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    print(token_embeds.shape,'token_embeds.shape')
    learned_embed1=torch.load(args.learned_embed_path1)#[args.placeholder_token1]
    learned_embed1=learned_embed1[args.placeholder_token1]
    initializer_token_ids = tokenizer.encode(args.prior_concept1, add_special_tokens=False)
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
    # std_pipeline = StableDiffusionPipelineClsAug.from_pretrained( model_name,
    #                         unet=accelerator.unwrap_model(unet, **extra_args),
    #                         tokenizer=accelerator.unwrap_model(tokenizer, **extra_args),
    #                         )
    # std_scheduler = std_pipeline.scheduler
    # std_fe_extractor = std_pipeline.feature_extractor
    # del std_pipeline
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
            scheduler=accelerator.unwrap_model(noise_scheduler, **extra_args),
            feature_extractor=None,
            safety_checker=None,
            requires_safety_checker=False,
        )
    if args.include_prior_concept:
        if args.rev:
            placeholder='{} {}'.format(args.prior_concept1, args.placeholder_token1)
        else:
            placeholder='{} {}'.format(args.placeholder_token1, args.prior_concept1)
    else:
        placeholder='{}'.format(args.placeholder_token1)
    if args.eval_prompt_type=='cat':
                eval_prompts = [
        "Photo of a {}.".format(placeholder),
        "{} swimming in a pool.".format(placeholder),
        "{} at a beach with a view of the seashore.".format(placeholder),
        "{} sitting on a window.".format(placeholder),
        "{} in times square.".format(placeholder),
        "{} is wearing sunglasses.".format(placeholder),
        "{} wearing a construction outfit.".format(placeholder),
        "{} is playing with a ball.".format(placeholder),
        "{} is wearing headphones.".format(placeholder),
        "{} oil painting ghibli inspired.".format(placeholder),
        "Painting of {} at a beach by artist claude monet.".format(placeholder),
        "{} digital painting 3d render geometric style.".format(placeholder),
        "Georgia O'Keeffe style {} painting.".format(placeholder),
        "a {} in chef outfit.".format(placeholder),
        "A {} is reading a book.".format(placeholder),
        "a {} is acting in a play wearing a costume.".format(placeholder),
        "a screaming {}.".format(placeholder),
        "a sleeping {}.".format(placeholder),
        "a sad {}.".format(placeholder),
        "a sculpture of the {}.".format(placeholder),
        ]*args.num_images_per_prompt
    elif args.eval_prompt_type=='dog':
        eval_prompts = [
            "Photo of a {}.".format(placeholder),
            "{} swimming in a pool.".format(placeholder),
            "{} at a beach with a view of the seashore.".format(placeholder),
            "{} in times square.".format(placeholder),
            "{} is wearing sunglasses.".format(placeholder),
            "{} is wearing a sombrero.".format(placeholder),
            "{} is in a construction outfit.".format(placeholder),
            "{} is playing with a ball.".format(placeholder),
            "{} is wearing headphones.".format(placeholder),
            "{} oil painting ghibli inspired.".format(placeholder),
            "Painting of {} at a beach by artist claude monet.".format(placeholder),
            "{} digital painting 3d render geometric style.".format(placeholder),
            "Georgia O'Keeffe style {} painting.".format(placeholder),
            "a {} in chef outfit.".format(placeholder),
            "A {} is reading a book.".format(placeholder),
            "a {} is acting in a play wearing a costume.".format(placeholder),
            "a barking {}.".format(placeholder),
            "a sleeping {}.".format(placeholder),
            "a sad {}.".format(placeholder),
            "a sculpture of {}.".format(placeholder),
            ]*args.num_images_per_prompt
    elif args.eval_prompt_type=='vase':
        eval_prompts = [
            "Photo of a {}.".format(placeholder),
            "{} in grand canyon.".format(placeholder),
            "{} with mountains and sunset in the background.".format(placeholder),
            "{} floating in a pool.".format(placeholder),
            "A wide shot of {} in times square.".format(placeholder),
            "{} and chocolate cake on a table.".format(placeholder),
            "Rose flowers in the {} on a table.".format(placeholder),
            "Marigold flowers in the {}.".format(placeholder),
            "The {} at the entrance to a medieval castle.".format(placeholder),
            "{} with pens in it.".format(placeholder),
            "{} oil painting ghibli inspired.".format(placeholder),
            "{} painting by artist claude monet.".format(placeholder),
            "A watercolor painting of {}.".format(placeholder),
            "A digital illustration of the {}.".format(placeholder),
            "Georgia O'Keeffe style {} painting.".format(placeholder),
            "A handbag in the style of {}.".format(placeholder),
            "A teapot in the style of {}.".format(placeholder),
            "{} made of stone.".format(placeholder),
            "A pair of {} on a study table.".format(placeholder),
            "A coffee cup in the style of the {} on a table.".format(placeholder),
            ]*args.num_images_per_prompt
    elif args.eval_prompt_type=='chair':
        eval_prompts=[
            "Photo of a {}.".format(placeholder),
            "{} near a pool.".format(placeholder),
            "{} at a beach with a view of the seashore.".format(placeholder),
            "{} in a garden.".format(placeholder),
            "{} in grand canyon.".format(placeholder),
            "{} in front of a medieval castle.".format(placeholder),
            "{} and a coffee table.".format(placeholder),
            "floor lamp on the side of {}.".format(placeholder),
            "{} and an orange sofa.".format(placeholder),
            "{} and a table with chocolate cake on it.".format(placeholder),
            "{} oil painting ghibli inspired.".format(placeholder),
            "{} painting by artist claude monet.".format(placeholder),
            "a watercolor painting of {} in a forest.".format(placeholder),
            "A digital illustration of the {}.".format(placeholder),
            "Georgia O'Keeffe style {} painting.".format(placeholder),
            "An orange {}.".format(placeholder),
            "A pink {}.".format(placeholder),
            "A red color {}.".format(placeholder),
            "{} crochet.".format(placeholder),
            "An egg chair in the style of {}.".format(placeholder),
            ]
    elif args.eval_prompt_type=='barn':
        eval_prompts=[
            "photo of a {}.".format(placeholder),
            "{} in snowy ice.".format(placeholder),
            "{} in the fall season with leaves all around.".format(placeholder),
            "{} at a beach with a view of the seashore.".format(placeholder),
            "Photo of the {} with the sun rising in the sky.".format(placeholder),
            "{} with forest in the background.".format(placeholder),
            "puppy in front of the {}.".format(placeholder),
            "cat sitting in front of the {}.".format(placeholder),
            "cat sitting in front of {} in snowy ice.".format(placeholder),
            "squirrel in front of the {}.".format(placeholder),
            "{} oil painting ghibli inspired.".format(placeholder),
            "{} painting by artist claude monet.".format(placeholder),
            "{} digital painting 3d render geometric style.".format(placeholder),
            "Georgia O'Keeffe style {} painting.".format(placeholder),
            "a watercolor painting of the {}.".format(placeholder),
            "painting of {} in the style of van gogh.".format(placeholder),
            "A futuristic {}.".format(placeholder),
            "A surreal landscape, {}.".format(placeholder),
            "A close up shot of the {}.".format(placeholder),
            "Top view of the {}.".format(placeholder),
            ]
    elif args.eval_prompt_type=='toy':
        eval_prompts=[
            "photo of a {}.".format(placeholder),
            "{} in grand canyon.".format(placeholder),
            "{} with mountains and sunset in the background.".format(placeholder),
            "{} floating in a pool.".format(placeholder),
            "A wide shot of {} in times square.".format(placeholder),
            "{} and a clock on a sofa.".format(placeholder),
            "A {} and a laptop on a study table.".format(placeholder),
            "A rusty {} in a post-apocalyptic landscape.".format(placeholder),
            "A {} and lego bricks lying on a rug.".format(placeholder),
            "An old {} lying on the sidewalk.".format(placeholder),
            "{} oil painting ghibli inspired.".format(placeholder),
            "{} painting by artist claude monet.".format(placeholder),
            "A watercolor painting of {}.".format(placeholder),
            "A digital illustration of {}.".format(placeholder),
            "Georgia O'Keeffe style {} painting.".format(placeholder),
            "{} made of black stone.".format(placeholder),
            "A pair of {} on a study table.".format(placeholder),
            "A {} made of colorful crystals and glass.".format(placeholder),
            "A house in the style of {}.".format(placeholder),
            "a backpack in the style of {}.".format(placeholder),
            ]*args.num_images_per_prompt
    elif args.eval_prompt_type=='flower':
        eval_prompts=[
            "Photo of a {}.".format(placeholder),
            "{} growing in the desert.".format(placeholder),
            "{} at a beach with a view of the seashore.".format(placeholder),
            "{} on top of a mountain with sunrise in the background.".format(placeholder),
            "A bouquet of {}.".format(placeholder),
            "A garden of {}.".format(placeholder),
            "{} in a violet vase on a table.".format(placeholder),
            "a vase filled with {} on a table.".format(placeholder),
            "Bouquet of {} and roses.".format(placeholder),
            "{} and a chocolate cake on the table.".format(placeholder),
            "{} oil painting ghibli inspired.".format(placeholder),
            "{} painting by artist claude monet.".format(placeholder),
            "A digital illustration of {}.".format(placeholder),
            "Georgia O'Keeffe style {} painting.".format(placeholder),
            "a watercolor painting of {} and a teapot on the table.".format(placeholder),
            "{} with violet color petals.".format(placeholder),
            "a sky blue color {}.".format(placeholder),
            "a beige colored {}.".format(placeholder),
            "A dried up {} in between a book.".format(placeholder),
            "a {} made of crystal.".format(placeholder),
            ]*args.num_images_per_prompt
    elif args.eval_prompt_type=='sunglasses':
        eval_prompts=[
        "photo of a {}.".format(placeholder),
        "{} on top of a vintage car in the countryside.".format(placeholder),
        "A pair of {} rest on a bookshelf.".format(placeholder),
        "{} lying on a study table.".format(placeholder),
        "close shot of {} on the sandy beach with a view of the seashore.".format(placeholder),
        "A futuristic robot wearing {}.".format(placeholder),
        "A person wearing {}.".format(placeholder),
        "A chef wearing {} prepares a gourmet meal.".format(placeholder),
        "A scientist wearing {} examines a test tube.".format(placeholder),
        "A dog wearing {} on the porch.".format(placeholder),
        "A giraffe wearing {}.".format(placeholder),
        "{} painted in the style of andy warhol.".format(placeholder),
        "painting of {} by artist claude monet.".format(placeholder),
        "A digital illustration of {}.".format(placeholder),
        "a modern art piece of {}.".format(placeholder),
        "a watercolor painting of {}.".format(placeholder),
        "cool neon party cat in {}.".format(placeholder),
        "digital painting of a turtle wearing {}.".format(placeholder),
        "smart hedgehog in {}.".format(placeholder),
        "{} digital 3d render.".format(placeholder),
        ]*args.num_images_per_prompt
    elif args.eval_prompt_type=='wooden_pot':
        eval_prompts=[
            "Photo of a {}.".format(placeholder),
            "{} in grand canyon.".format(placeholder),
            "{} with mountains and sunset in the background.".format(placeholder),
            "{} floating in a pool.".format(placeholder),
            "A wide shot of {} in times square.".format(placeholder),
            "{} and chocolate cake on a table.".format(placeholder),
            "Rose flowers in {} on a table.".format(placeholder),
            "Marigold flowers in the {}.".format(placeholder),
            "The {} at the entrance to a medieval castle.".format(placeholder),
            "{} with pens in it.".format(placeholder),
            "{} oil painting ghibli inspired.".format(placeholder),
            "{} painting by artist claude monet.".format(placeholder),
            "A watercolor painting of {}.".format(placeholder),
            "A digital illustration of the {}.".format(placeholder),
            "Georgia O'Keeffe style {} painting.".format(placeholder),
            "A handbag in the style of {}.".format(placeholder),
            "A teapot in the style of {}.".format(placeholder),
            "{} made of stone.".format(placeholder),
            "A pair of {} on a study table.".format(placeholder),
            "A coffee cup in the style of {} on a table.".format(placeholder),
            ]*args.num_images_per_prompt
    elif args.eval_prompt_type=='backpack':
        eval_prompts=[
            "photo of a {}.".format(placeholder),
            "{} on a rustic wooden dock overlooking a peaceful lake.",
            "{} on a café table with a steaming cup of coffee nearby.",
            "A {} on a trail in a pine forest.",
            "A {} on a bicycle parked in a tulip field.",
            "{} on the glass table of a high-rise penthouse overlooking a stunning city skyline.",
            "A {} with the night sky.",
            "A {} lying beside a bookshelf in a study room.",
            "A person is walking with {} in hand.",
            "A cat is looking from inside the {}.".format(placeholder),
            "{} is hanging from a wall hook.",
            "A wet {} in water.",
            "A graffiti stencil art of the {}.".format(placeholder),
            "A digital rendering of {}.".format(placeholder),
            "A Victorian-era oil painting of a {}.".format(placeholder),
            "A watercolor painting of {} on a stool.",
            "Modern art painting of a person with {}.".format(placeholder),
            "A purple colored {}.".format(placeholder),
            "A purple colored {} lying on a sofa.",
            "A cube shaped {}.".format(placeholder),
            ]*args.num_images_per_prompt
    elif args.eval_prompt_type=='teddybear':
        eval_prompts=[
            "Photo of a {}.".format(placeholder),
            "{} in grand canyon.".format(placeholder),
            "{} swimming in a pool.".format(placeholder),
            "{} sitting at the beach with a view of the sea.".format(placeholder),
            "{} in times square.".format(placeholder),
            "{} wearing sunglasses.".format(placeholder),
            "{} working on the laptop.".format(placeholder),
            "{} on a boat in the sea.".format(placeholder),
            "{} wearing headphones.".format(placeholder),
            "{} in a construction outfit.".format(placeholder),
            "{} oil painting ghibli inspired.".format(placeholder),
            "{} painting by artist claude monet.".format(placeholder),
            "A digital illustration of {}.".format(placeholder),
            "Georgia O'Keeffe style {} painting.".format(placeholder),
            "a watercolor painting of {} on top of a mountain.".format(placeholder),
            "A koala in the style of {}.".format(placeholder),
            "a backpack in the style of {}.".format(placeholder),
            "{} made of crochet.".format(placeholder),
            "A blanket in the style of {}.".format(placeholder),
            "a screaming {}.".format(placeholder),
            ]*args.num_images_per_prompt
    elif args.eval_prompt_type=='living':
        eval_prompts = [
            'a {} in the jungle'.format(placeholder),
            'a {} in the snow'.format(placeholder),
            'a {} on the beach'.format(placeholder),
            'a {} on a cobblestone street'.format(placeholder),
            'a {} on top of pink fabric'.format(placeholder),
            'a {} on top of a wooden floor'.format(placeholder),
            'a {} with a city in the background'.format(placeholder),
            'a {} with a mountain in the background'.format(placeholder),
            'a {} with a blue house in the background'.format(placeholder),
            'a {} on top of a purple rug in a forest'.format(placeholder),
            'a {} wearing a red hat'.format(placeholder),
            'a {} wearing a santa hat'.format(placeholder),
            'a {} wearing a rainbow scarf'.format(placeholder),
            'a {} wearing a black top hat and a monocle'.format(placeholder),
            'a {} in a chef outfit'.format(placeholder),
            'a {} in a firefighter outfit'.format(placeholder),
            'a {} in a police outfit'.format(placeholder),
            'a {} wearing pink glasses'.format(placeholder),
            'a {} wearing a yellow shirt'.format(placeholder),
            'a {} in a purple wizard outfit'.format(placeholder),
            'a red {}'.format(placeholder),
            'a purple {}'.format(placeholder),
            'a shiny {}'.format(placeholder),
            'a wet {}'.format(placeholder),
            'a cube shaped {}'.format(placeholder)
            ]*args.num_images_per_prompt
    elif args.eval_prompt_type=='nonliving':
        eval_prompts = [
            'a {} in the jungle'.format(placeholder),
            'a {} in the snow'.format(placeholder),
            'a {} on the beach'.format(placeholder),
            'a {} on a cobblestone street'.format(placeholder),
            'a {} on top of pink fabric'.format(placeholder),
            'a {} on top of a wooden floor'.format(placeholder),
            'a {} with a city in the background'.format(placeholder),
            'a {} with a mountain in the background'.format(placeholder),
            'a {} with a blue house in the background'.format(placeholder),
            'a {} on top of a purple rug in a forest'.format(placeholder),
            'a {} with a wheat field in the background'.format(placeholder),
            'a {} with a tree and autumn leaves in the background'.format(placeholder),
            'a {} with the Eiffel Tower in the background'.format(placeholder),
            'a {} floating on top of water'.format(placeholder),
            'a {} floating in an ocean of milk'.format(placeholder),
            'a {} on top of green grass with sunflowers around it'.format(placeholder),
            'a {} on top of a mirror'.format(placeholder),
            'a {} on top of the sidewalk in a crowded street'.format(placeholder),
            'a {} on top of a dirt road'.format(placeholder),
            'a {} on top of a white rug'.format(placeholder),
            'a red {}'.format(placeholder),
            'a purple {}'.format(placeholder),
            'a shiny {}'.format(placeholder),
            'a wet {}'.format(placeholder),
            'a cube shaped {}'.format(placeholder)
            ]*args.num_images_per_prompt
        
    else:
        assert False,args.eval_prompt_type
    # batch_size=len(prompts)
    batch_size=args.eval_batch_size
    num_batches=(len(eval_prompts)//batch_size)+int((len(eval_prompts)/batch_size)>0)

    # validation_files=os.listdir(args.train_data_dir1)
    # validation_target=Image.open(os.path.join((args.train_data_dir1),validation_files[0])).resize((512,512))
    count=0
    validation_files=sorted(os.listdir(args.train_data_dir1))
    validation_target=Image.open(os.path.join((args.train_data_dir1),validation_files[0])).resize((512,512)).convert('RGB')
    caption_data={}
    # print(learned_embed1.shape,'learned_embed1.shape')
    # print(prior_embed.shape,'prior_embed.shape')
    prior_embed=prior_embed.to(accelerator.device)
    learned_embed1=learned_embed1.to(accelerator.device)
    
    attn_mod_params={
                    'calibrate_ppos1':args.calibrate_ppos1,
                    'calibrate_ppos2':args.calibrate_ppos2,
                    'calibrate_pneg1':args.calibrate_pneg1,
                    'calibrate_pneg2':args.calibrate_pneg2,
                    'calibrate_kpos1':args.calibrate_kpos1,
                    'calibrate_kpos2':args.calibrate_kpos2,
                    'calibrate_kneg1':args.calibrate_kneg1,
                    'calibrate_kneg2':args.calibrate_kneg2,
                }

    with torch.no_grad():
        generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
        if args.normalize_target1:
            target_emb1=F.normalize(learned_embed1,p=1,dim=-1)*args.normalize_target1
        else:
            target_emb1=learned_embed1
        for batch_idx in range(num_batches):
            prompts=eval_prompts[batch_idx*batch_size:(batch_idx+1)*batch_size]
            if not len(prompts):
                break
            is_keyword_tokens_list=[]
            is_prior1_list=[]
            for prompt in prompts:
                is_keyword_tokens=[False]
                is_prior1=[False]
                text_words=prompt.split()
                for word_idx in range(len(text_words)):
                    cap_word=text_words[word_idx]
                    word_token_ids=tokenizer.encode(cap_word,add_special_tokens=False)
                    num_tokens=len(word_token_ids)
                    for tok_id in word_token_ids:
                        tok_decoded=tokenizer.decode(tok_id)
                        if args.placeholder_token1 == cap_word:
                            is_keyword_tokens.append(True)
                        else:
                            is_keyword_tokens.append(False)
                        # prior1
                        if tok_decoded==args.prior_concept1:
                            is_prior1.append(True)
                        else:
                            is_prior1.append(False)
                for _ in range(len(is_keyword_tokens),tokenizer.model_max_length):
                    is_keyword_tokens.append(False)
                for _ in range(len(is_prior1),tokenizer.model_max_length):
                    is_prior1.append(False)
                assert len(is_keyword_tokens)==tokenizer.model_max_length
                assert len(is_prior1)==tokenizer.model_max_length
                assert sum(is_keyword_tokens)==1
                assert sum(is_prior1)==1
                is_keyword_tokens=torch.BoolTensor(is_keyword_tokens)
                is_prior1=torch.BoolTensor(is_prior1)
                is_keyword_tokens_list.append(is_keyword_tokens)
                is_prior1_list.append(is_prior1)
            is_keyword_tokens_list=torch.stack(is_keyword_tokens_list)
            is_prior1_list=torch.stack(is_prior1_list)
            images = pipeline(prompt=prompts, 
                            num_inference_steps=50, 
                            guidance_scale=7.5, width=512, height=512,
                            num_images_per_prompt=1,
                            is_keyword_tokens1=is_keyword_tokens_list,
                            inj_embeddings1=target_emb1,
                            # is_prior1=is_prior1_list,
                            # attn_mod_params=attn_mod_params,
                            generator=generator,
                            add_pe=args.add_pe,
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
            torch.cuda.empty_cache()
            # 


            # merged_viz = Image.new('RGB', (512*(len(images)+1), 512), (255, 255, 255))
            # merged_viz.paste(validation_target,(0,0))
            # for iidx,(image, prompt) in enumerate(zip(images,prompts)):
            #     image_name='{:04d}'.format(count+1)
            #     img_path=os.path.join(sample_dir,'{}.jpg'.format(image_name))
            #     caption_data[image_name]=prompt.replace(placeholder,args.prior_concept1)
            #     merged_viz.paste(image.convert('RGB'),((iidx+1)*512,0))
            #     image.save(img_path)
            #     count+=1
            # print(batch_idx+1,num_batches)
            # merged_viz.save(os.path.join(merged_dir,'merged_{:03d}.jpg'.format(batch_idx+1)))
            # for iidx,prompt in enumerate(prompts):
            #     image_name='{:04d}'.format(count+1)
            #     caption_data[image_name]=prompt.replace(args.placeholder_token1,args.prior_concept1)
            #     count+=1
            #     print(count,len(eval_prompts))
            # break
    json.dump(caption_data,caption_file,indent=1)
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
    exit()
