

from utils import render_caption
import importlib
import sys
import os
sys.path.insert(0, './packages')
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))
from datasets_pkgs.dataset_cd import CustomDiffusionDataset
from configs import parse_args
import cv2
from data_utils import cycle, create_wbd
import argparse
import itertools
import json
import logging
import math
import random
import shutil
import warnings
from pathlib import Path

import numpy as np
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import HfApi, create_repo
# from huggingface_hub.utils import insecure_hashlib
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import (
    CustomDiffusionAttnProcessor,
    CustomDiffusionAttnProcessor2_0,
    CustomDiffusionXFormersAttnProcessor,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
torch.autograd.set_detect_anomaly(True)

# ADDED
import copy
import inspect
torch.use_deterministic_algorithms(True)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    print(worker_seed,'worker_seed')
    np.random.seed(worker_seed)
    random.seed(worker_seed)
# ADDED


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.30.0.dev0")

logger = get_logger(__name__)




def log_validation(
    text_encoder,
    tokenizer,
    unet,
    vae,
    args,
    accelerator,
    weight_dtype,
    global_step,
    pipeline,
):
    
    if args.include_prior_concept:
        placeholder='{} {}'.format(args.placeholder_token1,args.train_prior_concept1)
    else:
        placeholder='{}'.format(args.placeholder_token1)
    
    if args.eval_prompt_type=='living':
        validation_prompts=[
        'a {0} in the jungle'.format(placeholder),
        # 'a {0} with a city in the background'.format(placeholder),
        'a {0} with a mountain in the background'.format(placeholder),
        'a {0} on top of a purple rug in a forest'.format(placeholder),
        'a {0} in a chef outfit'.format(placeholder),
        # 'a {0} in a police outfit'.format(placeholder),
        'a cube shaped {0}'.format(placeholder)
        ]

    elif args.eval_prompt_type =='nonliving':
        validation_prompts = [
            'a {0} in the jungle'.format(placeholder),
            # 'a {0} with a city in the background'.format(placeholder),
            # 'a {0} with a mountain in the background'.format(placeholder),
            'a {0} with the Eiffel Tower in the background'.format(placeholder),
            'a {0} floating on top of water'.format(placeholder),
            # 'a {0} floating in an ocean of milk'.format(placeholder),
            'a {0} on top of the sidewalk in a crowded street'.format(placeholder),
            'a cube shaped {0}'.format(placeholder)
            ]
    else:
        assert False, 'undefined eval prompt type'
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        '\t'.join(validation_prompts)
    )

    # pipeline_args = {}
    # if vae is not None:
    #     pipeline_args["vae"] = vae
    # # create pipeline (note: unet and vae are loaded again in float32)
    # pipeline = DiffusionPipeline.from_pretrained(
    #     args.pretrained_model_name_or_path,
    #     tokenizer=tokenizer,
    #     text_encoder=text_encoder,
    #     unet=unet,
    #     revision=args.revision,
    #     variant=args.variant,
    #     feature_extractor=None,
    #     safety_checker=None,
    #     requires_safety_checker=False,
    #     torch_dtype=weight_dtype,
    #     **pipeline_args,
    # )

    # # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    # scheduler_args = {}

    # if "variance_type" in pipeline.scheduler.config:
    #     variance_type = pipeline.scheduler.config.variance_type

    #     if variance_type in ["learned", "learned_range"]:
    #         variance_type = "fixed_small"

    #     scheduler_args["variance_type"] = variance_type

    # module = importlib.import_module("diffusers")
    # scheduler_class = getattr(module, args.validation_scheduler)
    # pipeline.scheduler = scheduler_class.from_config(pipeline.scheduler.config, **scheduler_args)
    # pipeline = pipeline.to(accelerator.device)
    # pipeline.set_progress_bar_config(disable=True)

    # if args.pre_compute_text_embeddings:
    #     pipeline_args = {
    #         "prompt_embeds": prompt_embeds,
    #         "negative_prompt_embeds": negative_prompt_embeds,
    #     }
    # else:
    # pipeline_args = {"prompt": args.validation_prompt}

    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    images = []
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)
    with autocast_ctx:
        with torch.no_grad():
            torch.cuda.empty_cache()
            images = pipeline(validation_prompts, 
                            num_inference_steps=25, 
                            generator=generator,
                            ).images
        print('Generated')
    # for tracker in accelerator.trackers:
    #     if tracker.name == "tensorboard":
    #         np_images = np.stack([np.asarray(img) for img in images])
    #         tracker.writer.add_images("validation", np_images, global_step, dataformats="NHWC")
    #     if tracker.name == "wandb":
    #         tracker.log(
    #             {
    #                 "validation": [
    #                     wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
    #                 ]
    #             }
    #         )
    del pipeline
    torch.cuda.empty_cache()

    return images,validation_prompts
def freeze_params(params):
    for param in params:
        param.requires_grad = False

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")




def collate_fn(examples,with_prior_preservation):
    if 'pixel_values' in examples[0]:
        # 1. pixel_values
        pixel_values = [example["pixel_values"] for example in examples]
        masks = [example["mask"] for example in examples]
        # 2. input ids
        input_ids = [example["input_ids"] for example in examples]
        # 3. prior preseravation
        is_keyword_tokens = [example["is_keyword_tokens"] for example in examples] #N,77, list of booleans
        if with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            is_keyword_tokens += [example["is_keyword_tokens_prior"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]
            masks += [example["class_mask"] for example in examples]
        is_keyword_tokens = torch.stack(is_keyword_tokens)
        input_ids=torch.stack(input_ids)
        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        raw_captions_ti = [example["raw_caption_ti"] for example in examples]
        masks = torch.stack(masks)
        masks = masks.to(memory_format=torch.contiguous_format).float().unsqueeze(1)


        # 5. For MLM 
        input_ids_masked = []
        input_ids_pos = []
        masked_idxs = []
        mlm_labels = []
        non_special_idxs = []
        is_keyword_tokens_mlm = []
        raw_captions_mlm = []

    else:
        # FOR TI
        pixel_values=[]
        input_ids=[]
        is_keyword_tokens=[]
        masks=[]
        raw_captions_mlm = [example["raw_caption_mlm"] for example in examples]
        raw_captions_ti = []
        # FOR TI

    
        # 5. For MLM 
        input_ids_masked = [example["input_ids_masked"] for example in examples]
        input_ids_masked=torch.stack(input_ids_masked)
        input_ids_pos = [example["input_ids_pos"] for example in examples]
        input_ids_pos=torch.stack(input_ids_pos)
        masked_idxs = [example["masked_idxs"] for example in examples] #N,77, list of booleans
        masked_idxs = torch.stack(masked_idxs)
        mlm_labels = [example["mlm_labels"] for example in examples] #N,77, list of booleans
        mlm_labels = torch.stack(mlm_labels)
        non_special_idxs = [example["non_special_idxs"] for example in examples] #N,77, list of booleans
        non_special_idxs = torch.stack(non_special_idxs)
        is_keyword_tokens_mlm = [example["is_keyword_tokens_mlm"] for example in examples] #N,77, list of booleans
        is_keyword_tokens_mlm = torch.stack(is_keyword_tokens_mlm)
        # 5. For MLM 
    batch = {
        "masks": masks,
        "pixel_values": pixel_values,
        "input_ids": input_ids, # for reconstruction
        "input_ids_masked": input_ids_masked, # for mlm
        "input_ids_pos": input_ids_pos, # for mlm
        "masked_idxs": masked_idxs,
        "mlm_labels": mlm_labels,
        "non_special_idxs": non_special_idxs,
        "is_keyword_tokens_mlm": is_keyword_tokens_mlm,
        "is_keyword_tokens": is_keyword_tokens,
        "raw_captions_ti": raw_captions_ti,
        "raw_captions_mlm": raw_captions_mlm,
    }
    return batch




def save_new_embed(text_encoder, placeholder_token_id1, accelerator, args, output_dir, safe_serialization=False):
    """Saves the new token embeddings from the text encoder."""
    logger.info("Saving embeddings")
    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight
    learned_embeds_dict = {}
    learned_embeds_dict[args.placeholder_token1] = learned_embeds[placeholder_token_id1]
    filename = f"{output_dir}/{args.placeholder_token1}.bin"
    if safe_serialization:
        safetensors.torch.save_file(learned_embeds_dict, filename, metadata={"format": "pt"})
    else:
        torch.save(learned_embeds_dict, filename)



def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=None)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
        

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    # if accelerator.is_main_process:
    #     accelerator.init_trackers("custom-diffusion", config=vars(args))

    # If passed along, set the training seed now.
    # if args.seed is not None:
    #     set_seed(args.seed)
    if args.seed is not None:
        print('set seed',args.seed)
        # set_seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        exp_dir=os.path.join(args.output_dir,args.run_name)  
        viz_dir = os.path.join(exp_dir,'viz')
        os.makedirs(viz_dir, exist_ok=True)
        codepath=os.path.join(exp_dir,'src')
        if os.path.exists(codepath) and 'tmp' not in args.run_name:
            assert False
        os.makedirs(codepath,exist_ok=True)
        os.system('cp *.py {}'.format(codepath))
        os.system('cp ../datasets_pkgs {} -R'.format(codepath))
        os.system('cp packages {} -R'.format(codepath))

        caption_log_path=os.path.join(codepath,'log_captions.txt')
        caption_log_file=open(caption_log_path,'w')

        # copy clip
        os.makedirs(os.path.join(codepath,'clip_src'),exist_ok=True)
        target = os.readlink('clip_src/modeling_clip.py')
        shutil.copy2(target, '{}/clip_src/modeling_clip.py'.format(codepath))
        target = os.readlink('clip_src/modeling_outputs.py')
        shutil.copy2(target, '{}/clip_src/modeling_outputs.py'.format(codepath))
        # copy clip
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


    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
            use_fast=False,
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    # Adding a modifier token which is optimized ####
    # Code taken from https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py
    initializer_token_id = []
    placeholder_tokens = [args.placeholder_token1]
    # `mask_tokens = [args.mask_tokens]
    # tokenizer.add_tokens(mask_tokens)
    tokenizer.add_tokens(placeholder_tokens)
    if args.lambda_mlm:
        mask_tokens = [args.mask_tokens]
        tokenizer.add_tokens(mask_tokens)
        mask_token_ids = tokenizer.convert_tokens_to_ids(mask_tokens)
    else:
        mask_token_ids=None
    # mask_token_ids = tokenizer.convert_tokens_to_ids(mask_tokens)
    placeholder_token_id1 = tokenizer.convert_tokens_to_ids(placeholder_tokens)
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    if args.lambda_mlm and args.mask_embed_path is not None:
        mask_embeds=torch.load(args.mask_embed_path)[args.mask_tokens].to(accelerator.device)
        mask_embeds_initial=mask_embeds.clone().detach()
        with torch.no_grad():
            for token_id in mask_token_ids:
                token_embeds[token_id] = mask_embeds

    # FOR TI
    if args.initializer_token and args.resume_cd_path is None:
        initializer_token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
        initial_embed=token_embeds[initializer_token_ids].clone().to(accelerator.device)
        with torch.no_grad():
            for token_id in placeholder_token_id1:
                token_embeds[token_id] = initial_embed.clone()
        print(args.initializer_token,'initializer_token')
        print(initial_embed.shape,'initial_embed.shape')
    # with torch.no_grad():
    #     for token_id in mask_token_ids:
    #         token_embeds[token_id] = mask_embeds.clone()
    #     if args.initialize_token:
    #         initializer_token_ids = tokenizer.encode(args.prior_concept1, add_special_tokens=False)
    #         initializer_token_id = initializer_token_ids[0]
    #         prior_embed=token_embeds[initializer_token_id].detach().clone().unsqueeze(0)
    #         for token_id in placeholder_token_id1:
    #             token_embeds[token_id] = token_embeds[initializer_token_id].clone()
    #         prior_embed=prior_embed.to(accelerator.device)
    # Add learned concept
    # if args.resume_path is not None and args.resume_path!='None':
    #     learned_embed_path1=os.path.join(args.resume_path,'{}.bin'.format(args.placeholder_token1))
    #     learned_embed1=torch.load(learned_embed_path1)#[args.placeholder_token]
    #     learned_embed1=learned_embed1[args.placeholder_token1].to('cpu')
    #     print(token_embeds[placeholder_token_id1].shape,'token_embeds[placeholder_token_id1].shape')
    #     print(learned_embed1.shape,'learned_embed1.shape')
    #     with torch.no_grad():
    #         token_embeds[placeholder_token_id1] = learned_embed1.clone()
    #     print('load ti embeddings')
    #     del learned_embed1
    if args.lambda_mlm:
        if 'contextnetv5' in args.cls_net_path:
            from contextnet_v3 import ContextNetV3
            if 'stable-diffusion-2-1' in args.pretrained_model_name_or_path:
                cls_net=ContextNetV3(1024, len(token_embeds)-1) #-1 for placeholder
                cls_output_dim=len(token_embeds)-1
            elif 'stable-diffusion-v1-5' in args.pretrained_model_name_or_path:
                if 'mlm_contextnet_' in args.cls_net_path:
                    cls_net=ContextNetV3(768, len(token_embeds)) # -1 for placeholder
                    cls_output_dim=len(token_embeds)
                else:
                    cls_net=ContextNetV3(768, len(token_embeds)-1) # -1 for placeholder
                    cls_output_dim=len(token_embeds)-1
        else:
            from contextnet import ContextNet
            if 'stable-diffusion-2-1' in args.pretrained_model_name_or_path:
                cls_net=ContextNet(1024, len(token_embeds)-1) #-1 for placeholder
                cls_output_dim=len(token_embeds)-1
            elif 'stable-diffusion-v1-5' in args.pretrained_model_name_or_path:
                if 'mlm_contextnet_' in args.cls_net_path:
                    cls_net=ContextNet(768, len(token_embeds)) # -1 for placeholder
                    cls_output_dim=len(token_embeds)
                else:
                    cls_net=ContextNet(768, len(token_embeds)-1) # -1 for placeholder
                    cls_output_dim=len(token_embeds)-1
            else:
                assert False,'undefined sd version'

    # Freeze all parameters except for the token embeddings in text encoder
    params_to_freeze = itertools.chain(
            text_encoder.text_model.encoder.parameters(),
            text_encoder.text_model.final_layer_norm.parameters(),
            text_encoder.text_model.embeddings.position_embedding.parameters(),
        )
    freeze_params(params_to_freeze)
    # if not args.train_text_encoder:
        
    # else:
    #     # params_to_freeze = itertools.chain(
    #     #     # text_encoder.text_model.encoder.parameters(),
    #     #     # text_encoder.text_model.final_layer_norm.parameters(),
    #     #     text_encoder.text_model.embeddings.position_embedding.parameters(),
    #     # )
    #     # freeze_params(params_to_freeze)
    #     text_encoder.requires_grad_(True)
    # if args.placeholder_token1 is None:
    #     text_encoder.requires_grad_(False)
    # for key, val in text_encoder.named_parameters():
    #     if val.requires_grad:
    #         print(key)
    # exit()

    ########################################################
    ########################################################
    
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    vae.requires_grad_(False)
    
    unet.requires_grad_(False)
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    if accelerator.mixed_precision != "fp16" and args.placeholder_token1 is not None:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    # if not args.train_text_encoder and text_encoder is not None:
    #     text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    attention_class = (
        CustomDiffusionAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else CustomDiffusionAttnProcessor
    )
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            attention_class = CustomDiffusionXFormersAttnProcessor
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    if not args.train_text_encoder:
        # assert args.learned_embed_path1,args.learned_embed_path1
        # text_encoder.requires_grad_(False)
        text_encoder.text_model.encoder.requires_grad_(False)
        text_encoder.text_model.final_layer_norm.requires_grad_(False)
        text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
    for key, val in text_encoder.named_parameters():
        print(val.requires_grad,key,'text_encoder requires_grad')
        if 'token_embedding' in key:
            assert val.requires_grad
            
    # now we will add new Custom Diffusion weights to the attention layers
    # It's important to realize here how many attention weights will be added and of which sizes
    # The sizes of the attention layers consist only of two different variables:
    # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
    # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

    # Let's first see how many attention processors we will have to set.
    # For Stable Diffusion, it should be equal to:
    # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
    # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
    # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
    # => 32 layers

    # Only train key, value projection layers if freeze_model = 'crossattn_kv' else train all params in the cross attention layer
    train_kv = True
    train_q_out = False if args.freeze_model == "crossattn_kv" else True
    custom_diffusion_attn_procs = {}
    st = unet.state_dict()
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
            ).to(unet.device)
            custom_diffusion_attn_procs[name].load_state_dict(weights)
        else:
            custom_diffusion_attn_procs[name] = attention_class(
                train_kv=False,
                train_q_out=False,
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
            )
    del st
    unet.set_attn_processor(custom_diffusion_attn_procs)
    custom_diffusion_layers = AttnProcsLayers(unet.attn_processors)
    tmp_state_dict=custom_diffusion_layers.state_dict()
    if accelerator.is_main_process:
        for key in tmp_state_dict:
            # layer_name = f"Layer {i}"
            print(key,'layer_name')
            assert 'custom_diffusion' in key
    del tmp_state_dict
    accelerator.register_for_checkpointing(custom_diffusion_layers)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.resume_cd_path is not None:
            text_encoder.gradient_checkpointing_enable()
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        if args.with_prior_preservation:
            args.learning_rate = args.learning_rate * 2.0

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    if args.resume_cd_path is not None:
        params_to_optimize = [
            {"params": text_encoder.get_input_embeddings().parameters(), "lr": args.learning_rate},
        ]
    else:
        if args.train_text_encoder:
            params_to_optimize = [
                {"params": text_encoder.parameters(), "lr": args.learning_rate},
                {"params": custom_diffusion_layers.parameters(), "lr": args.learning_rate},
            ]
        else:
            params_to_optimize = [
                {"params": custom_diffusion_layers.parameters(), "lr": args.learning_rate},
                {"params": text_encoder.get_input_embeddings().parameters(), "lr": args.learning_rate},
            ]
    
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    # Dataset and DataLoaders creation:
    if args.exclude_cap_types is not None:
        exclude_cap_types=args.exclude_cap_types.split('-')
    else:
        exclude_cap_types=None
    train_dataset = CustomDiffusionDataset(
        get_images=True,
        include_prior_concept=args.include_prior_concept,
        data_root=args.train_data_dir1,
        tokenizer=tokenizer,
        size=args.resolution,
        placeholder_token=(" ".join(tokenizer.convert_ids_to_tokens(placeholder_token_id1))),
        center_crop=args.center_crop,
        flip_p=args.flip_p,
        train_prior_concept1=args.train_prior_concept1,
        mask_token_ids=mask_token_ids,
        mlm_target=args.mlm_target,
        prompt_type=args.train_prompt_type,
        class_data_root=args.class_data_dir1 if args.with_prior_preservation else None,
        class_num=args.num_class_images,
        class_prompt=args.class_prompt1,
        simple_caption=args.simple_caption,
        mask_prob=args.mask_prob,
        seed=args.seed,
        exclude_cap_types=exclude_cap_types,
        caption_root=args.caption_root,
        aug=not(args.noaug),
    )
    train_dataset_mlm = CustomDiffusionDataset(
        get_images=False,
        include_prior_concept=args.include_prior_concept,
        data_root=args.train_data_dir1,
        tokenizer=tokenizer,
        size=args.resolution,
        placeholder_token=(" ".join(tokenizer.convert_ids_to_tokens(placeholder_token_id1))),
        center_crop=args.center_crop,
        flip_p=args.flip_p,
        train_prior_concept1=args.train_prior_concept1,
        mask_token_ids=mask_token_ids,
        mlm_target=args.mlm_target,
        prompt_type=args.train_prompt_type,
        mask_prob=args.mask_prob,
        simple_caption=args.simple_caption,
        seed=args.seed,
        exclude_cap_types=exclude_cap_types,
        caption_root=args.caption_root,
        aug=not(args.noaug),
    )
    generator = torch.Generator(device='cpu').manual_seed(args.seed)
    generator_cuda = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, 
        shuffle=True, 
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        generator=generator,
        num_workers=args.dataloader_num_workers,
        worker_init_fn=seed_worker,
    )
    # HERE
    mlm_loader = torch.utils.data.DataLoader(
            train_dataset_mlm,
            batch_size=args.mlm_batch_size,
            shuffle=True,
            collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
            num_workers=args.dataloader_num_workers,
            generator=generator,
            worker_init_fn=seed_worker,
        )
    mlm_loader = cycle(mlm_loader)
    def load_mlm_batch(mlm_loader):
        mlm_data=next(mlm_loader)
        return mlm_data

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
    )

    # Prepare everything with our `accelerator`.
    if args.lambda_mlm:
        if args.train_text_encoder:
            custom_diffusion_layers, text_encoder, optimizer, train_dataloader, lr_scheduler, cls_net, mlm_loader = accelerator.prepare(
            custom_diffusion_layers, text_encoder, optimizer, train_dataloader, lr_scheduler, cls_net, mlm_loader
        )
        else:
            custom_diffusion_layers, text_encoder, optimizer, train_dataloader, lr_scheduler, cls_net, mlm_loader  = accelerator.prepare(
            custom_diffusion_layers, text_encoder, optimizer, train_dataloader, lr_scheduler, cls_net, mlm_loader
        )
    
    else: # NO MLM
        if args.train_text_encoder:
            custom_diffusion_layers, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            custom_diffusion_layers, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
        else:
            custom_diffusion_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                custom_diffusion_layers, optimizer, train_dataloader, lr_scheduler
            )
    # Potentially load in the weights and states from a previous save
    if args.resume_cd_path and args.resume_cd_path!='None':
        cd_layers_path=os.path.join(args.resume_cd_path,'custom_diffusion.pt')
        saved_state_dict = torch.load(cd_layers_path, map_location=torch.device('cpu'))
        for key in saved_state_dict:
            print(key,'saved')
        print()
        defined_state_dict=unet.state_dict()
        new_state_dict={}
        for key in defined_state_dict:
            print(key,'defined')
            if key in saved_state_dict:
                new_state_dict[key]=saved_state_dict[key]
            else:
                new_state_dict[key]=defined_state_dict[key]
        unet.load_state_dict(new_state_dict,strict=True)
        print('unet parameters loaded')
        del new_state_dict
    # if not args.train_text_encoder and text_encoder is not None:
    #     text_encoder.to(accelerator.device, dtype=weight_dtype)
    # ADDED
    if args.lambda_mlm and args.cls_net_path is not None:
        for defined_key in cls_net.state_dict():
            print(defined_key,'defined_key-clsnet')
        saved_state_dict = torch.load(args.cls_net_path, map_location=torch.device('cpu'))
        new_state_dict={}
        for saved_key in saved_state_dict:
            new_key=saved_key
            print(saved_key,'saved_key-clsnet')
            if accelerator.num_processes>1:
                if not saved_key.startswith('module.'):
                    new_key='module.'+saved_key
            else:
                if saved_key.startswith('module.'):
                    new_key=saved_key.replace('module.','')
            new_state_dict[new_key]=saved_state_dict[saved_key]
        cls_net.load_state_dict(new_state_dict,strict=True)
    # ADDED


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Processes = {accelerator.num_processes}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # PIPELINE
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
    print(accepts_keep_fp32_wrapper,'accepts_keep_fp32_wrapper')
    print(extra_args,'extra_args')
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
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=False)
    # PIPELINE
    
    # if args.resume_from_checkpoint:
    #     if args.resume_from_checkpoint != "latest":
    #         path = os.path.basename(args.resume_from_checkpoint)
    #     else:
    #         # Get the most recent checkpoint
    #         dirs = os.listdir(args.output_dir)
    #         dirs = [d for d in dirs if d.startswith("checkpoint")]
    #         dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    #         path = dirs[-1] if len(dirs) > 0 else None
    #     if path is None:
    #         accelerator.print(
    #             f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
    #         )
    #         args.resume_from_checkpoint = None
    #         initial_global_step = 0
    #     else:
    #         accelerator.print(f"Resuming from checkpoint {path}")
    #         accelerator.load_state(os.path.join(args.output_dir, path))
    #         global_step = int(path.split("-")[1])
    #         initial_global_step = global_step
    #         first_epoch = global_step // num_update_steps_per_epoch
    # else:
    #     initial_global_step = 0

    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()
    cos_sim=torch.nn.CosineSimilarity(dim=-1, eps=1e-08)
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.placeholder_token1 is not None:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet), accelerator.accumulate(text_encoder):
                # Load Batch
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                input_ids=batch["input_ids"]# B,77 list of booleans (tensor)
                is_keyword_tokens=batch["is_keyword_tokens"]# B,77 list of booleans (tensor)
                masks=batch["masks"]# B,77 list of booleans (tensor)
                raw_captions_ti=batch["raw_captions_ti"]
                # masks64=torch.nn.functional.interpolate(masks,(64,64))
                
                # Load Batch




                # Convert images to latent space
                latents = vae.encode(pixel_values.to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)



                learned_embeds=accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[min(placeholder_token_id1) : max(placeholder_token_id1) + 1]
                if args.normalize_target1:
                    target_emb=F.normalize(learned_embeds,p=1,dim=-1)*args.normalize_target1
                else:
                    target_emb=learned_embeds
                encoder_hidden_states = text_encoder(
                                                    input_ids,
                                                    # is_keyword_tokens1=is_keyword_tokens,
                                                    # inj_embeddings1=target_emb,
                                                     )[0].to(dtype=weight_dtype)
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)
                    masks = torch.chunk(batch["masks"], 2, dim=0)[0]
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = ((loss * masks).sum([1, 2, 3]) / masks.sum([1, 2, 3])).mean()
                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    masks = batch["masks"]
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = ((loss * masks).sum([1, 2, 3]) / masks.sum([1, 2, 3])).mean()

                # 3. MLM Loss
                loss_mlm=None
                if args.lambda_mlm:
                    # for MLM
                    batch_mlm=load_mlm_batch(mlm_loader)
                    is_keyword_tokens_mlm=batch_mlm["is_keyword_tokens_mlm"]
                    masked_idxs=batch_mlm["masked_idxs"]
                    mlm_labels=batch_mlm["mlm_labels"].to(accelerator.device)
                    non_special_idxs=batch_mlm["non_special_idxs"]
                    input_ids_masked=batch_mlm["input_ids_masked"].to(accelerator.device)
                    input_ids_pos=batch_mlm["input_ids_pos"].to(accelerator.device)
                    raw_captions_mlm=batch_mlm["raw_captions_mlm"]
                    # for MLM
                    clip_text_embedding_masked = text_encoder(input_ids_masked,
                                                            # mask_embedding=mask_embeds.unsqueeze(0),
                                                            # mask_idxs=masked_idxs,
                                                            # is_keyword_tokens1=is_keyword_tokens_mlm,
                                                            # inj_embeddings1=target_emb,
                                                            )[0].to(accelerator.device, dtype=weight_dtype)
                    # clip_text_embedding_masked = text_encoder(input_ids_masked)[0].to(accelerator.device, dtype=weight_dtype)
                    mlm_logits=cls_net(clip_text_embedding_masked)
                    # masked_idxs_flat=masked_idxs.view(-1)
                    loss_mlm = F.cross_entropy(
                        mlm_logits.view(-1,cls_output_dim),
                        mlm_labels.view(-1),
                        ignore_index=-100,
                        reduction='mean'
                    )
                    # loss_mlm[masked_idxs_flat]*=args.mlm_weight
                    # loss_mlm=loss_mlm.mean()
                    loss+=(loss_mlm*args.lambda_mlm)

                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        # itertools.chain(text_encoder.parameters(), custom_diffusion_layers.parameters())
                        # if args.placeholder_token1 is not None
                        # else custom_diffusion_layers.parameters()
                        itertools.chain(text_encoder.parameters(), custom_diffusion_layers.parameters())
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                # Token Embeddings
                if args.train_text_encoder: 
                    # # update everything
                    pass
                    
                else:
                    # no train then do not update token embeddings
                    # except the placeholder
                    index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
                    assert isinstance(placeholder_token_id1,list)
                    updated_ids=copy.deepcopy(placeholder_token_id1)
                    index_no_updates[min(updated_ids) : max(updated_ids) + 1] = False #everything except placeholder
                    with torch.no_grad():
                        accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                            index_no_updates] = orig_embeds_params[index_no_updates]
                if args.lambda_mlm and args.freeze_mask_embedding:
                    with torch.no_grad():
                        accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                            mask_token_ids
                        ] = orig_embeds_params[mask_token_ids]
                        assert len(orig_embeds_params[mask_token_ids])==1

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # [1] CHECKPOINTING
                if (global_step % args.checkpointing_steps == 0 or global_step==1):
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
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

                        save_path = os.path.join(ckpt_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path,exist_ok=True)
                        unet = unet.to(torch.float32)
                        learned_embeds_saved=accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[min(placeholder_token_id1) : max(placeholder_token_id1) + 1]
                        learned_embeds_dict = {args.placeholder_token1: learned_embeds_saved.cpu()}
                        weight_name = "learned_embeds_s{:04d}.pt".format(global_step)
                        save_path_emb = os.path.join(ckpt_dir, f"checkpoint-{global_step}/learned_embeds.pt")
                        torch.save(learned_embeds_dict, save_path_emb)
                        del learned_embeds_saved

                        if args.resume_cd_path is None: 
                            save_path_unet = os.path.join(ckpt_dir, f"checkpoint-{global_step}/custom_diffusion.pt")
                            cur_state_dict=unet.state_dict()
                            save_state_dict={}
                            for key in cur_state_dict:
                                if 'custom_diffusion' in key:
                                    save_state_dict[key]=cur_state_dict[key]
                            torch.save(save_state_dict,save_path_unet)
                            if args.train_text_encoder:
                                save_path_text_encoder = os.path.join(ckpt_dir, f"checkpoint-{global_step}/text_encoder.pt")
                                torch.save(text_encoder.state_dict(),save_path_text_encoder)
                # [1] CHECKPOINTING



                # CAPTION LOGGGING/ INPUT LOGGING / MLM LOGGING / VALIDATION
                if accelerator.is_main_process:
                    # [2] CAPTION LOGGING
                    if ((global_step % args.log_steps == 0) or global_step==1):
                        caption_log_file=open(caption_log_path,'a')
                        for raw_caption_ti in raw_captions_ti:
                            caption_log_file.write('STEP{:04d}\t{}\n'.format(global_step,raw_caption_ti))
                            caption_log_file.flush()
                        caption_log_file.write('\n')
                        caption_log_file.flush()
                        if args.lambda_mlm:
                            for raw_caption_mlm in raw_captions_mlm:
                                caption_log_file.write('STEP{:04d}\t{}\n'.format(global_step,raw_caption_mlm))
                                caption_log_file.flush()
                            caption_log_file.write('\n')
                        caption_log_file.write('\n')
                        caption_log_file.flush()
                        caption_log_file.close()
                    # [2] CAPTION LOGGING
                    

                    if (global_step % args.validation_steps == 0  or global_step==1):
                        # [3] INPUT LOGGING
                        input_image=(pixel_values[0].permute(1,2,0).detach().cpu().numpy()+1)*127.5
                        input_mask=masks[0].permute(1,2,0).detach().cpu().numpy()
                        input_image=input_image.astype(np.uint8)
                        input_image=Image.fromarray(input_image)
                        input_image.save(os.path.join(viz_dir,'input_image_s{:05d}.jpg'.format(global_step)))
                        # [3] INPUT LOGGING


                        # [4] MLM LOGGING
                        if args.lambda_mlm:
                            # 1. MLM Result Logging
                            viz_idx=0
                            masked_idxs=masked_idxs.detach().cpu().numpy()[viz_idx:viz_idx+1]
                            non_special_idxs=non_special_idxs.detach().cpu()[viz_idx:viz_idx+1]
                            mlm_logits=mlm_logits.argmax(-1).detach().cpu().numpy()[viz_idx:viz_idx+1]#1,77
                            input_ids_pos=input_ids_pos[viz_idx:viz_idx+1]
                            input_ids_masked=input_ids_masked[viz_idx:viz_idx+1]

                            input_ids_pos=input_ids_pos[non_special_idxs]
                            input_ids_masked=input_ids_masked[non_special_idxs]
                            mlm_logits=mlm_logits[non_special_idxs]
                            masked_idxs=masked_idxs[non_special_idxs]

                            decoded=tokenizer.batch_decode(input_ids_pos)
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
                        # [4] MLM LOGGING

                        # [5] VALIDTION
                        images,validation_prompts = log_validation(
                                # unwrap_model(text_encoder) if text_encoder is not None else text_encoder,
                                accelerator.unwrap_model(text_encoder),
                                tokenizer,
                                unwrap_model(unet),
                                vae,
                                args,
                                accelerator,
                                weight_dtype,
                                global_step,
                                pipeline=pipeline,
                            )
                        validation_files=sorted(os.listdir(args.train_data_dir1))
                        validation_target=Image.open(os.path.join((args.train_data_dir1),validation_files[0])).resize((512,512)).convert('RGB')
                        num_images=len(images)
                        num_cols=num_images
                        num_rows=num_images//num_cols
                        margin_bottom=150
                        margin_right=10
                        merged_viz = Image.new('RGB', ((512+margin_right)*(num_cols+1), (512+margin_bottom)*num_rows), (255, 255, 255))
                        for ridx in range(num_rows):
                            merged_viz.paste(validation_target,(0,ridx*(512+margin_bottom)))
                        for iidx,(image, val_prompt) in enumerate(zip(images[:],validation_prompts[:])):
                            row_idx=iidx//num_cols
                            col_idx=iidx-(num_cols*row_idx)
                            x0=(col_idx+1)*(512+margin_right)
                            y0=row_idx*(512+margin_bottom)+512
                            x1=x0+(512+margin_right)
                            y1=y0+margin_bottom
                            merged_viz=render_caption(merged_viz,val_prompt,[x0,y0+20,x1,y1])
                            merged_viz.paste(image.convert('RGB'),((col_idx+1)*(512+margin_right),row_idx*(512+margin_bottom)))
                        merged_viz.save(os.path.join(sample_dir, 'sample_{:05d}.jpg'.format(global_step)))
                        # [5] VALIDTION
            # sync_grad        
            # [6] PBAR PRINTING
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            with torch.no_grad():
                target_embeds_log=accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[min(placeholder_token_id1) : max(placeholder_token_id1) + 1].clone()
                norm_target=torch.norm(target_embeds_log,p=1,dim=-1)
                logs['sim_target']=cos_sim(target_embeds_log,initial_embed.detach()).item()
                logs['same_target']=bool(torch.all(target_embeds_log==initial_embed).item())
                del target_embeds_log

            if loss_mlm is not None:
                mask_embeds_log = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[min(mask_token_ids) : max(mask_token_ids) + 1].clone()
                logs['loss_mlm']=loss_mlm.detach().item()
                norm_mask=torch.norm(mask_embeds,p=1,dim=-1)
                logs['norm_mask']=norm_mask.item()
                logs['sim_mask']=cos_sim(mask_embeds_log,mask_embeds_initial.detach()).item()
                logs['same_mask']=bool(torch.all(mask_embeds_log==mask_embeds_initial).item())
                del mask_embeds_log
            logs['norm_target']=norm_target.item()
            progress_bar.set_postfix(**logs)
            # [6] PBAR PRINTING
            
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break
            #ft
    # Save the custom diffusion layers
    accelerator.wait_for_everyone()
    
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)