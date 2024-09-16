import random
import os
import inspect

from contextlib import nullcontext
from configs import parse_args
from utils import render_caption
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.insert(0, './packages')
from datasets_pkgs.dataset_db_clean import DreamboothDataset
from data_utils import cycle, create_wbd
import argparse
import copy
import gc
import importlib
import itertools
import logging
import math
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, model_info, upload_folder
from huggingface_hub.utils import insecure_hashlib
from packaging import version
from PIL import Image
from PIL.ImageOps import exif_transpose
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
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
# from diffusers.training_utils import compute_snr
from diffusers.utils import is_wandb_available
# from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

if is_wandb_available():
    import wandb
# ADDED
torch.autograd.set_detect_anomaly(True)
torch.use_deterministic_algorithms(True)
# ADDED
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.29.0.dev0")

logger = get_logger(__name__)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    print(worker_seed,'worker_seed')
    np.random.seed(worker_seed)
    random.seed(worker_seed)
print(torch.cuda.is_available(),'avail')


def log_validation(
    text_encoder,
    tokenizer,
    unet,
    vae,
    args,
    accelerator,
    weight_dtype,
    global_step,
    target_emb,
    generator,
    pipeline,
):
    
    if args.include_prior_concept:
        placeholder='{} {}'.format(args.placeholder_token1,args.train_prior_concept1)
    else:
        placeholder='{}'.format(args.placeholder_token1)
    

    if args.eval_prompt_type=='living':
        validation_prompts=[
        # 'a {0} in the jungle'.format(placeholder),
        # 'a {0} with a city in the background'.format(placeholder),
        'a {0} with a mountain in the background'.format(placeholder),
        'a {0} on top of a purple rug in a forest'.format(placeholder),
        'a {0} in a chef outfit'.format(placeholder),
        # 'a {0} in a police outfit'.format(placeholder),
        # 'a cube shaped {0}'.format(placeholder)
        ]
    elif args.eval_prompt_type =='nonliving':
        validation_prompts = [
            # 'a {0} in the jungle'.format(placeholder),
            # 'a {0} with a city in the background'.format(placeholder),
            # 'a {0} with a mountain in the background'.format(placeholder),
            'a {0} with the Eiffel Tower in the background'.format(placeholder),
            'a {0} floating on top of water'.format(placeholder),
            # 'a {0} floating in an ocean of milk'.format(placeholder),
            'a {0} on top of the sidewalk in a crowded street'.format(placeholder),
            # 'a cube shaped {0}'.format(placeholder)
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
    # create pipeline (note: unet and vae are loaded again in float32)
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
    # noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    # accepts_keep_fp32_wrapper = "keep_fp32_wrapper" in set(
    #     inspect.signature(
    #         accelerator.unwrap_model
    #     ).parameters.keys()
    # )
    # extra_args = (
    #     {"keep_fp32_wrapper": True}
    #     if accepts_keep_fp32_wrapper
    #     else {}
    # )
    # unet=unet.to(accelerator.device)
    # pipeline = StableDiffusionPipeline(
    #         vae=accelerator.unwrap_model(vae, **extra_args),
    #         unet=accelerator.unwrap_model(unet, **extra_args),
    #         text_encoder=accelerator.unwrap_model(text_encoder, **extra_args),
    #         tokenizer=accelerator.unwrap_model(tokenizer, **extra_args),
    #         scheduler=accelerator.unwrap_model(noise_scheduler, **extra_args),
    #         feature_extractor=None,
    #         safety_checker=None,
    #         requires_safety_checker=False,
    #     )

    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
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
    # pipeline_args = {"prompt": args.validation_prompt}
    
    
    
    
    # run inference
    # generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    images = []
    # if args.validation_images is None:
    #     with torch.autocast("cuda"):
    #         images = pipeline(prompt=validation_prompts,
    #                             num_inference_steps=25, 
    #                             generator=generator).images[0]
    # else:
    #     for image in args.validation_images:
    #         image = Image.open(image)
    #         image = pipeline(prompt=validation_prompts, image=image, generator=generator).images[0]
    #         images.append(image)

    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)
    with autocast_ctx:
        with torch.no_grad():
            images = pipeline(
                        validation_prompts, 
                            num_inference_steps=25, 
                            generator=generator,
                            verbose=True,
                            ).images
        print('Generated')
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, global_step, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    torch.cuda.empty_cache()

    return images,validation_prompts


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
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")







def collate_fn(examples,with_prior_preservation=False):
        if 'pixel_values' in examples[0]:
            # 1. pixel_values
            pixel_values = [example["pixel_values"] for example in examples]
            # 2. input ids
            input_ids = [example["input_ids"] for example in examples]
            
            # 3. prior preseravation
            if with_prior_preservation:
                input_ids += [example["class_prompt_ids"] for example in examples]
                pixel_values += [example["class_images"] for example in examples]
            input_ids=torch.stack(input_ids)
            pixel_values = torch.stack(pixel_values)
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            raw_captions_ti = [example["raw_caption_ti"] for example in examples]
            # 5. For MLM 
            input_ids_masked = []
            input_ids_pos = []
            masked_idxs = []
            mlm_labels = []
            non_special_idxs = []
            raw_captions_mlm = []
            
        else:
            pixel_values=[]
            input_ids=[]
            # masks=[]
            raw_captions_mlm = [example["raw_caption_mlm"] for example in examples]
            raw_captions_ti = []
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
            # 5. For MLM 
       
        


        batch = {
            "pixel_values": pixel_values,
            "input_ids": input_ids, # for reconstruction
            "input_ids_masked": input_ids_masked, # for mlm
            "input_ids_pos": input_ids_pos, # for mlm
            "masked_idxs": masked_idxs,
            "mlm_labels": mlm_labels,
            "non_special_idxs": non_special_idxs,
            # "masks": masks,
            "raw_captions_mlm": raw_captions_mlm,
            "raw_captions_ti": raw_captions_ti,
        }
        return batch


class PromptDataset(Dataset):
    """A simple dataset to prepare the prompts to generate class images on multiple GPUs."""

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def model_has_vae(args):
    config_file_name = Path("vae", AutoencoderKL.config_name).as_posix()
    if os.path.isdir(args.pretrained_model_name_or_path):
        config_file_name = os.path.join(args.pretrained_model_name_or_path, config_file_name)
        return os.path.isfile(config_file_name)
    else:
        files_in_repo = model_info(args.pretrained_model_name_or_path, revision=args.revision).siblings
        return any(file.rfilename == config_file_name for file in files_in_repo)


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
        return_dict=False,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    # logging_dir = Path(args.output_dir, args.logging_dir)
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

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

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
    # if args.seed is not None:
    #     set_seed(args.seed)
    if args.seed is not None:
        print('set seed',args.seed)
        # random.seed(args.seed)
        # np.random.seed(args.seed)
        # torch.manual_seed(args.seed)
        # torch.cuda.manual_seed_all(args.seed)
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
        

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
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

    # if model_has_vae(args):
    #     vae = AutoencoderKL.from_pretrained(
    #         args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    #     )
    # else:
    #     vae = None
    vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )


    # HERE
    placeholder_tokens = [args.placeholder_token1]
    tokenizer.add_tokens(placeholder_tokens)
    if args.lambda_mlm:
        mask_tokens = [args.mask_tokens]
        tokenizer.add_tokens(mask_tokens)
        mask_token_ids = tokenizer.convert_tokens_to_ids(mask_tokens)
    else:
        mask_token_ids=None
    placeholder_token_id1 = tokenizer.convert_tokens_to_ids(placeholder_tokens)
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    # mask_embeds=token_embeds[mask_token_ids]
    if args.lambda_mlm and args.mask_embed_path is not None:
        mask_embeds=torch.load(args.mask_embed_path)[args.mask_tokens].to(accelerator.device)
        mask_embeds_initial=mask_embeds.clone().detach()
        with torch.no_grad():
            for token_id in mask_token_ids:
                token_embeds[token_id] = mask_embeds
    
    if args.initializer_token and args.resume_unet_path is None:
        initializer_token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
        initial_embed=token_embeds[initializer_token_ids].clone().to(accelerator.device)
        with torch.no_grad():
            for token_id in placeholder_token_id1:
                token_embeds[token_id] = initial_embed.clone()
        print(args.initializer_token,'initializer_token')
        print(initial_embed.shape,'initial_embed.shape')
    # Add learned concept
    # if args.learned_embed_path1:
    #     learned_embed1=torch.load(args.learned_embed_path1)#[args.placeholder_token]
    #     print('load ti embeddings')
    #     learned_embed1=learned_embed1[args.placeholder_token1]
    #     # initial_embed=learned_embed1.clone().detach()
    #     with torch.no_grad():
    #         token_embeds[placeholder_token_id1] = learned_embed1.clone()
    #     del learned_embed1
    if args.lambda_mlm:
        if 'contextnetv6' in args.cls_net_path:
            from contextnet_v3 import ContextNetV3 as ContextNet
        else:
            from contextnet import ContextNet
        if 'stable-diffusion-2-1' in args.pretrained_model_name_or_path:
            hidden_dim=1024
        elif 'stable-diffusion-v1-5' in args.pretrained_model_name_or_path:
            hidden_dim=768
        cls_output_dim=len(token_embeds)-1
        cls_net=ContextNet(hidden_dim, cls_output_dim) # -1 for placeholder
    # HERE
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                sub_dir = "unet" if isinstance(model, type(unwrap_model(unet))) else "text_encoder"
                model.save_pretrained(os.path.join(output_dir, sub_dir))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        while len(models) > 0:
            # pop models so that they are not loaded again
            model = models.pop()

            if isinstance(model, type(unwrap_model(text_encoder))):
                # load transformers style into model
                load_model = text_encoder_cls.from_pretrained(input_dir, subfolder="text_encoder")
                model.config = load_model.config
            else:
                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if vae is not None:
        vae.requires_grad_(False)

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

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            print('xformer')
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()
        

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )

    if unwrap_model(unet).dtype != torch.float32:
        raise ValueError(f"Unet loaded as datatype {unwrap_model(unet).dtype}. {low_precision_error_string}")

    if args.train_text_encoder and unwrap_model(text_encoder).dtype != torch.float32:
        raise ValueError(
            f"Text encoder loaded as datatype {unwrap_model(text_encoder).dtype}." f" {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

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

    # Optimizer creation
    if args.resume_unet_path is not None:
        params_to_optimize = [
                {"params": text_encoder.get_input_embeddings().parameters(), "lr": args.learning_rate},
                ]
    else:
        if args.train_text_encoder:
            params_to_optimize = [
                {"params": text_encoder.parameters(), "lr": args.learning_rate},
                {"params": unet.parameters(), "lr": args.learning_rate},
                # {"params": text_encoder.get_input_embeddings().parameters(), "lr": args.learning_rate},
                ]
        else:
            params_to_optimize = [
                {"params": unet.parameters(), "lr": args.learning_rate},
                {"params": text_encoder.get_input_embeddings().parameters(), "lr": args.learning_rate},
                ]
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    

    
    print(mask_token_ids,'mask_token_ids')
    if args.exclude_cap_types is not None:
        exclude_cap_types=args.exclude_cap_types.split('-')
    else:
        exclude_cap_types=None
    train_dataset = DreamboothDataset(
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
        get_images=True,
        prompt_type=args.train_prompt_type,
        class_data_root=args.class_data_dir1 if args.with_prior_preservation else None,
        class_num=args.num_class_images,
        class_prompt=args.class_prompt1,
        simple_caption=args.simple_caption,
        mask_prob=args.mask_prob,
        seed=args.seed,
        exclude_cap_types=exclude_cap_types,
        caption_root=args.caption_root,
    )
    train_dataset_mlm = DreamboothDataset(
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
        get_images=False,
        prompt_type=args.train_prompt_type,
        mask_prob=args.mask_prob,
        simple_caption=args.simple_caption,
        seed=args.seed,
        exclude_cap_types=exclude_cap_types,
        caption_root=args.caption_root,
    )
    generator = torch.Generator(device='cpu').manual_seed(args.seed)
    generator_cuda = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, 
        shuffle=True, num_workers=args.dataloader_num_workers,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        generator=generator,
        worker_init_fn=seed_worker,
    )
    # HERE
    mlm_loader = torch.utils.data.DataLoader(
            train_dataset_mlm,
            batch_size=args.mlm_batch_size,
            shuffle=True,
            collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
            num_workers=args.dataloader_num_workers*2,
            generator=generator,
            worker_init_fn=seed_worker,
        )
    mlm_loader = cycle(mlm_loader)
    def load_mlm_batch(mlm_loader):
        mlm_data=next(mlm_loader)
        return mlm_data
    # HERE

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
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.lambda_mlm:
        if args.train_text_encoder:
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler, cls_net, mlm_loader = accelerator.prepare(
                unet, text_encoder, optimizer, train_dataloader, lr_scheduler, cls_net, mlm_loader
            )
        else: #for TI Fintuning
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler, cls_net, mlm_loader = accelerator.prepare(
                unet, text_encoder, optimizer, train_dataloader, lr_scheduler, cls_net, mlm_loader
            )
    else:# NO MLM
        if args.train_text_encoder:
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, text_encoder, optimizer, train_dataloader, lr_scheduler
            )
        else: #no_mlm, no text_encoder - never happens
            unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, optimizer, train_dataloader, lr_scheduler
            )
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
    if args.resume_unet_path and args.resume_unet_path!='None':
        state_dict = torch.load(args.resume_unet_path, map_location=torch.device('cpu'))
        # if not isinstance(state_dict,OrderedDict):
        #     state_dict=state_dict()
        unet.load_state_dict(state_dict,strict=True)
        print('unet parameters loaded')
        del state_dict
    if args.resume_text_encoder_path and args.resume_text_encoder_path!='None':
        state_dict = torch.load(args.resume_text_encoder_path, map_location=torch.device('cpu'))
        # if not isinstance(state_dict,OrderedDict):
        #     state_dict=state_dict()
        text_encoder.load_state_dict(state_dict,strict=True)
        initial_embed=accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[min(placeholder_token_id1) : max(placeholder_token_id1) + 1].clone().detach()
        print('text_encoder parameters loaded')
        del state_dict



    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to weight_dtype
    if vae is not None:
        vae.to(accelerator.device, dtype=weight_dtype)

    # if not args.train_text_encoder and text_encoder is not None:
    #     text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        tracker_config.pop("validation_images")
        accelerator.init_trackers("dreambooth", config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
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
    # pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=False)
    # PIPELINE

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
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
        if args.train_text_encoder:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Load Batch
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                input_ids=batch["input_ids"]# B,77 list of booleans (tensor)
                raw_captions_ti=batch["raw_captions_ti"]
                
                # Load Batch
                model_input = vae.encode(pixel_values.to(dtype=weight_dtype)).latent_dist.sample()
                model_input = model_input * vae.config.scaling_factor
                noise = torch.randn_like(model_input)
                bsz, channels, height, width = model_input.shape
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                )
                timesteps = timesteps.long()
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
                # learned_embeds=accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[min(placeholder_token_id1) : max(placeholder_token_id1) + 1]
                encoder_hidden_states = text_encoder(
                                                    input_ids,
                                                     )[0].to(dtype=weight_dtype)
                # if unwrap_model(unet).config.in_channels == channels * 2:
                #     noisy_model_input = torch.cat([noisy_model_input, noisy_model_input], dim=1)
                if args.class_labels_conditioning == "timesteps":
                    class_labels = timesteps
                else:
                    class_labels = None

                # Predict the noise residual
                model_pred = unet(
                    noisy_model_input, 
                    timesteps, 
                    encoder_hidden_states, 
                    class_labels=class_labels, 
                    return_dict=False
                )[0]
                # print(model_pred.shape,'model_pred.shape')
                # if model_pred.shape[1] == 6:
                #     model_pred, _ = torch.chunk(model_pred, 2, dim=1)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)
                    # Compute prior loss
                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")


                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                if args.with_prior_preservation:
                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss

                # 3. MLM Loss
                loss_mlm=None
                if args.lambda_mlm:
                    # for MLM
                    batch_mlm=load_mlm_batch(mlm_loader)
                    masked_idxs=batch_mlm["masked_idxs"]
                    mlm_labels=batch_mlm["mlm_labels"].to(accelerator.device)
                    non_special_idxs=batch_mlm["non_special_idxs"]
                    input_ids_masked=batch_mlm["input_ids_masked"].to(accelerator.device)
                    input_ids_pos=batch_mlm["input_ids_pos"].to(accelerator.device)
                    raw_captions_mlm=batch_mlm["raw_captions_mlm"]
                    # for MLM
                    clip_text_embedding_masked = text_encoder(input_ids_masked,
                                                            )[0].to(accelerator.device, dtype=weight_dtype)
                    mlm_logits=cls_net(clip_text_embedding_masked)
                    # masked_idxs_flat=masked_idxs.view(-1)
                    # loss_mlm = F.cross_entropy(
                    #     mlm_logits.view(-1,cls_output_dim),
                    #     mlm_labels.view(-1),
                    #     ignore_index=-100,
                    #     reduction='mean'
                    # )
                    mlm_labels_flat=mlm_labels.view(-1)
                    loss_mlm = F.cross_entropy(
                        mlm_logits.view(-1,cls_output_dim),
                        mlm_labels.view(-1),
                        # ignore_index=-100,
                        reduction='none'
                    )
                    loss_mlm=loss_mlm[mlm_labels_flat!=(-100)]
                    loss_mlm=loss_mlm.mean()
                    loss=loss+(loss_mlm*args.lambda_mlm)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if args.train_text_encoder
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                # del learned_embeds



                # Token Embeddings
                if args.train_text_encoder: 
                    # # update everything
                    # index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
                    # assert isinstance(placeholder_token_id1,list)
                    # updated_ids=copy.deepcopy(placeholder_token_id1)
                    # if args.lambda_mlm and args.freeze_mask_embedding==0:
                    #     updated_ids+=mask_token_ids
                    # index_no_updates[min(placeholder_token_id1) : max(placeholder_token_id1) + 1] = False #everything except placeholder
                    # with torch.no_grad():
                    #     accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                    #         index_no_updates
                    #     ] = orig_embeds_params[index_no_updates]
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
                # Token Embeddings

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                # [1] CHECKPOINTING
                if (global_step % args.checkpointing_steps == 0 or global_step==1) and not args.debug:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
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

                    save_dir = os.path.join(ckpt_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_dir,exist_ok=True)
                    
                    if args.resume_unet_path is None:
                        save_path_unet=os.path.join(save_dir,'unet_s{:04d}.pt'.format(global_step))
                        torch.save(unet.state_dict(),save_path_unet)
                        save_path_text_encoder=os.path.join(save_dir,'text_encoder_s{:04d}.pt'.format(global_step))
                        torch.save(text_encoder.state_dict(),save_path_text_encoder)
                    else: #ti training
                        learned_embeds_saved=accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[min(placeholder_token_id1) : max(placeholder_token_id1) + 1]
                        learned_embeds_dict = {args.placeholder_token1: learned_embeds_saved.cpu()}
                        weight_name = "learned_embeds_s{:04d}.pt".format(global_step)
                        save_path = os.path.join(save_dir, weight_name)
                        torch.save(learned_embeds_dict, save_path)
                        del learned_embeds_saved
                    logger.info(f"Saved state to {save_dir}")
                # [1] CHECKPOINTING



                # CAPTION LOGGGING/ INPUT LOGGING / MLM LOGGING / VALIDATION
                if accelerator.is_main_process:
                    # [2] CAPTION LOGGING
                    if ((global_step % args.log_steps == 0) or global_step==1) and accelerator.is_main_process:
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
                            special_idxs=~non_special_idxs
                            mlm_logits=mlm_logits.argmax(-1).detach().cpu().numpy()[viz_idx:viz_idx+1]#1,77
                            input_ids_pos=input_ids_pos[viz_idx:viz_idx+1]
                            input_ids_masked=input_ids_masked[viz_idx:viz_idx+1]

                            input_ids_pos=input_ids_pos[non_special_idxs]
                            input_ids_masked=input_ids_masked[non_special_idxs]
                            mlm_logits=mlm_logits[non_special_idxs]
                            masked_idxs=masked_idxs[non_special_idxs]


                            assert torch.all(mlm_labels[special_idxs]==(-100)),'mlm_label special_idx==-100'
                            mlm_labels=mlm_labels[non_special_idxs].detach().cpu().numpy()
                            mlm_labels=mlm_labels[mlm_labels>0]


                            decoded=tokenizer.batch_decode(input_ids_pos)
                            decoded_masked=tokenizer.batch_decode(input_ids_masked)
                            decoded_logits=tokenizer.batch_decode(mlm_logits)
                            decoded_labels=tokenizer.batch_decode(mlm_labels)
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
                            decoded_labels=' '.join(decoded_labels)
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
                            print('Labels\t\t|{}'.format(decoded_labels))
                            print(dots)
                            print()
                        # [4] MLM LOGGING

                        # [5] VALIDTION
                        with torch.no_grad():
                            learned_embeds_val=accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[min(placeholder_token_id1) : max(placeholder_token_id1) + 1]
                            images,validation_prompts = log_validation(
                                unwrap_model(text_encoder) if text_encoder is not None else text_encoder,
                                tokenizer,
                                unwrap_model(unet),
                                vae,
                                args,
                                accelerator,
                                weight_dtype,
                                global_step,
                                learned_embeds_val.clone(),
                                generator=generator_cuda,
                                pipeline=pipeline
                            )
                            del learned_embeds_val
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
                target_embeds_log = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[min(placeholder_token_id1) : max(placeholder_token_id1) + 1].clone()
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

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pipeline_args = {}

        if text_encoder is not None:
            pipeline_args["text_encoder"] = unwrap_model(text_encoder)

        if args.skip_save_text_encoder:
            pipeline_args["text_encoder"] = None

        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unwrap_model(unet),
            revision=args.revision,
            variant=args.variant,
            **pipeline_args,
            feature_extractor=None,
        safety_checker=None,
        requires_safety_checker=False,
        )

        # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
        scheduler_args = {}

        if "variance_type" in pipeline.scheduler.config:
            variance_type = pipeline.scheduler.config.variance_type

            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"

            scheduler_args["variance_type"] = variance_type

        pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config, **scheduler_args)

    

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
