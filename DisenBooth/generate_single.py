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
from diffusers.loaders import (
    LoraLoaderMixin,
    text_encoder_lora_state_dict,
)
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    SlicedAttnAddedKVProcessor,
)
from diffusers.models.lora import LoRALinearLayer
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
        set_seed(args.seed)

    exp_name=args.resume_unet_path.split('/')[-4]
    if not args.exp_dir:
        exp_dir=os.path.join(args.output_dir,exp_name)
    else:
        exp_dir=args.exp_dir
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
    mask_tokens = [args.mask_tokens]
    placeholder_token1 = [args.placeholder_token1]

    # placeholder_token2 = [args.placeholder_token2]
    tokenizer.add_tokens(mask_tokens)
    tokenizer.add_tokens(placeholder_token1)
    text_encoder.resize_token_embeddings(len(tokenizer))
    placeholder_token_id1 = tokenizer.convert_tokens_to_ids(placeholder_token1)
    if args.learned_embed_path1 is not None and args.learned_embed_path1 !='None':
        learned_embed1=torch.load(args.learned_embed_path1)#[args.placeholder_token]
        learned_embed1=learned_embed1[args.placeholder_token1]
        if args.normalize_target1:
            target_emb=F.normalize(learned_embed1,p=1,dim=-1)*args.normalize_target1
        else:
            target_emb=learned_embed1
        print('load ti embeddings')
        with torch.no_grad():
            text_encoder.get_input_embeddings().weight[placeholder_token_id1] = target_emb.clone()
        del learned_embed1
    

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

    # Set correct lora layers
    unet_lora_parameters = []
    for attn_processor_name, attn_processor in unet.attn_processors.items():
        # Parse the attention module.
        attn_module = unet
        for n in attn_processor_name.split(".")[:-1]:
            attn_module = getattr(attn_module, n)

        # Set the `lora_layer` attribute of the attention-related matrices.
        attn_module.to_q.set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_q.in_features, out_features=attn_module.to_q.out_features, rank=args.rank
            )
        )
        attn_module.to_k.set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_k.in_features, out_features=attn_module.to_k.out_features, rank=args.rank
            )
        )
        attn_module.to_v.set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_v.in_features, out_features=attn_module.to_v.out_features, rank=args.rank
            )
        )
        attn_module.to_out[0].set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_out[0].in_features,
                out_features=attn_module.to_out[0].out_features,
                rank=args.rank,
            )
        )

        # Accumulate the LoRA params to optimize.
        unet_lora_parameters.extend(attn_module.to_q.lora_layer.parameters())
        unet_lora_parameters.extend(attn_module.to_k.lora_layer.parameters())
        unet_lora_parameters.extend(attn_module.to_v.lora_layer.parameters())
        unet_lora_parameters.extend(attn_module.to_out[0].lora_layer.parameters())

        if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
            attn_module.add_k_proj.set_lora_layer(
                LoRALinearLayer(
                    in_features=attn_module.add_k_proj.in_features,
                    out_features=attn_module.add_k_proj.out_features,
                    rank=args.rank,
                )
            )
            attn_module.add_v_proj.set_lora_layer(
                LoRALinearLayer(
                    in_features=attn_module.add_v_proj.in_features,
                    out_features=attn_module.add_v_proj.out_features,
                    rank=args.rank,
                )
            )
            unet_lora_parameters.extend(attn_module.add_k_proj.lora_layer.parameters())
            unet_lora_parameters.extend(attn_module.add_v_proj.lora_layer.parameters())
    if args.resume_text_encoder_path:
        text_lora_parameters = LoraLoaderMixin._modify_text_encoder(text_encoder, dtype=torch.float32, rank=args.rank)

    """UNet Initialization"""
    print(inspect.getsourcefile(UNet2DConditionModel.from_pretrained), 'inspect')
    # for param in unet.parameters():
    #     param.requires_grad = False
    # vae.requires_grad_(False)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    # vae.to(accelerator.device, dtype=weight_dtype)
    # text_encoder.to(accelerator.device, dtype=weight_dtype)
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
    # if args.resume_tokenizer_path and args.resume_tokenizer_path!='None':
    #     state_dict = torch.load(args.resume_tokenizer_path, map_location=torch.device('cpu'))
    #     if not isinstance(state_dict,OrderedDict):
    #         state_dict=state_dict()
    #     tokenizer.load_state_dict(state_dict,strict=True)
    #     print('tokenizer parameters loaded')
    #     del state_dict
    if args.resume_unet_path and args.resume_unet_path!='None':
        state_dict = torch.load(args.resume_unet_path, map_location=torch.device('cpu'))
        if not isinstance(state_dict,OrderedDict):
            state_dict=state_dict()
        unet.load_state_dict(state_dict,strict=True)
        print('unet parameters loaded')
        del state_dict
    # if args.resume_text_encoder_path and args.resume_text_encoder_path!='None':
    #     state_dict = torch.load(args.resume_text_encoder_path, map_location=torch.device('cpu'))
    #     if not isinstance(state_dict,OrderedDict):
    #         state_dict=state_dict()
    #     text_encoder.load_state_dict(state_dict,strict=True)
    #     print('text_encoder parameters loaded')
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
    # model_id = "stabilityai/stable-diffusion-2-1-base"
    # pipeline = StableDiffusionPipeline.from_pretrained(model_id).to("cuda")
    # pipeline.load_lora_weights(args.resume_lora_path)
    print('lora loaded')
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

    # validation_files=os.listdir(args.train_data_dir1)
    # validation_target=Image.open(os.path.join((args.train_data_dir1),validation_files[0])).resize((512,512))
    count=0
    validation_files=sorted(os.listdir(args.train_data_dir1))
    validation_target=Image.open(os.path.join((args.train_data_dir1),validation_files[0])).resize((512,512)).convert('RGB')
    caption_data={}
    with torch.no_grad():
        learned_embeds=accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[min(placeholder_token_id1) : max(placeholder_token_id1) + 1]
        
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
            torch.cuda.empty_cache()
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
