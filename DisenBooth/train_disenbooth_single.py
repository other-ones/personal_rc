import inspect
from data_utils import cycle, create_wbd
from utils import render_caption
from datasets_pkgs.dataset_mlm import TextualInversionDataset
from configs import parse_args
import sys
sys.path.insert(0, './packages')
import argparse
import copy
import gc
import hashlib
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
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import open_clip
from disen_net import Image_adapter, cal_cos
import torch.nn as nn

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.utils.torch_utils import is_compiled_module

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
from diffusers.optimization import get_scheduler
from diffusers.training_utils import unet_lora_state_dict
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.

logger = get_logger(__name__)


def log_validation(tokenizer, args, accelerator, target_emb,pipeline,step):
    
    # create pipeline (note: unet and vae are loaded again in float32)
    

    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    # dog
    if args.include_prior_concept:
        placeholder='{} {}'.format(args.placeholder_token1,args.prior_concept1)
    else:
        placeholder='{}'.format(args.placeholder_token1)

    if args.prompt_type=='pet':
        validation_prompts=[
            "a picture of {} swimming in a pool".format(placeholder),
            "a picture of {} with the Great Wall of China in the background".format(placeholder),
            "a picture of {} in times square".format(placeholder),
            "{} on a boat in the sea".format(placeholder),
            "{} in a purple wizard outfit".format(placeholder),
            "{} playing with a ball".format(placeholder),
            "{} wearing sunglasses".format(placeholder),
            ]
    # vase
    
    elif args.prompt_type in ['nonliving']:
        validation_prompts = [
            'a {0} in the jungle'.format(placeholder),
            'a {0} in the snow'.format(placeholder),
            'a {0} with a blue house in the background'.format(placeholder),
            'a {0} with the Eiffel Tower in the background'.format(placeholder),
            'a purple {0}'.format(placeholder),
            'a wet {0}'.format(placeholder),
            'a cube shaped {0}'.format(placeholder)
            ]
    elif args.prompt_type in ['building']:
        validation_prompts = [
            '{} in snowy ice.'.format(placeholder),
            '{} at a beach with a view of the seashore.'.format(placeholder),
            'Photo of the {} with the sun rising in the sky.'.format(placeholder),
            'cat sitting in front of {} in snowy ice.'.format(placeholder),
            '{} digital painting 3d render geometric style.'.format(placeholder),
            'painting of {} in the style of van gogh.'.format(placeholder),
            'Top view of the {}. '.format(placeholder)
            ]
    elif args.prompt_type in ['sunglasses']:
        validation_prompts=[
            'photo of a {}'.format(placeholder),
            'close shot of {} on the sandy beach with a view of the seashore'.format(placeholder),
            'A scientist wearing {} examines a test tube'.format(placeholder),
            'A dog wearing {} on the porch'.format(placeholder),
            'A giraffe wearing {}'.format(placeholder),
            '{} painted in the style of andy warhol'.format(placeholder),
            'digital painting of a turtle wearing {}'.format(placeholder),
            '{} digital 3d render'.format(placeholder),
        ]
    else:
        assert False
    # print(validation_prompts[0],'validation_prompts')
    # print('Start Inference')
    is_keyword_tokens_list1=[]
    for prompt in validation_prompts:
        is_keyword_tokens1=[False]
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
        for _ in range(len(is_keyword_tokens1),tokenizer.model_max_length):
            is_keyword_tokens1.append(False)
        assert len(is_keyword_tokens1)==tokenizer.model_max_length
        is_keyword_tokens1=torch.BoolTensor(is_keyword_tokens1)
        is_keyword_tokens_list1.append(is_keyword_tokens1)
    is_keyword_tokens_list1=torch.stack(is_keyword_tokens_list1)
    logger.info(
        f"STEP {step} Running validation... \n Generating {len(validation_prompts)} images with prompt:"
        f" {validation_prompts}.",main_process_only=True
    )
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)
    with autocast_ctx:
        if args.lambda_mlm:
            images = pipeline(validation_prompts, num_inference_steps=25, generator=generator,
                            silent=args.silent,
                            inj_embeddings1=target_emb,
                            #   width=512, height=512, 
                            is_keyword_tokens1=is_keyword_tokens_list1).images
        else:
            images = pipeline(validation_prompts, num_inference_steps=25, generator=generator,
                            silent=args.silent,
                            #   width=512, height=512, 
                            ).images
    print('Generated')


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
            
            masks = [example["masks"] for example in examples]
            masks = torch.stack(masks)
            masks = masks.to(memory_format=torch.contiguous_format).float()

            # 2. input ids
            input_ids = [example["input_ids"] for example in examples]
            
            # 2. prior preseravation
            is_keyword_tokens = [example["is_keyword_tokens"] for example in examples] #N,77, list of booleans
            if with_prior_preservation:
                input_ids += [example["class_prompt_ids"] for example in examples]
                is_keyword_tokens += [example["is_keyword_tokens_prior"] for example in examples]
                pixel_values += [example["class_images"] for example in examples]
            is_keyword_tokens = torch.stack(is_keyword_tokens)
            input_ids=torch.stack(input_ids)
            pixel_values = torch.stack(pixel_values)
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        else:
            pixel_values=[]
            input_ids=[]
            is_keyword_tokens=[]
            masks=[]

       
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
            "pixel_values": pixel_values,
            "input_ids": input_ids, # for reconstruction
            "input_ids_masked": input_ids_masked, # for mlm
            "input_ids_pos": input_ids_pos, # for mlm
            "masked_idxs": masked_idxs,
            "mlm_labels": mlm_labels,
            "non_special_idxs": non_special_idxs,
            "is_keyword_tokens_mlm": is_keyword_tokens_mlm,
            "is_keyword_tokens": is_keyword_tokens,
            "masks": masks,
        }
        return batch


        



def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


def main(args):
    exp_dir=os.path.join(args.output_dir,args.run_name)    
    logging_dir = Path(exp_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=exp_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (sayakpaul): Remove this check when gradient accumulation with two models is enabled in accelerate.
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
    if args.seed is not None:
        set_seed(args.seed)
    # Handle the repository creation
    if accelerator.is_main_process:
        viz_dir = os.path.join(exp_dir,'viz')
        os.makedirs(viz_dir, exist_ok=True)
        codepath=os.path.join(exp_dir,'src')
        if os.path.exists(codepath) and 'tmp' not in codepath:
            assert False
        os.makedirs(codepath,exist_ok=True)
        os.system('cp *.py {}'.format(codepath))
        os.system('cp datasets_pkgs {} -R'.format(codepath))
        os.system('cp packages {} -R'.format(codepath))
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
    # Generate class images if prior preservation is enabled.

    # Handle the repository creation
    # if accelerator.is_main_process:
    #     if args.output_dir is not None:
    #         os.makedirs(args.output_dir, exist_ok=True)

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
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    try:
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
        )
    except OSError:
        # IF does not have a VAE so let's just set it to None
        # We don't have to error out here
        vae = None

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    
    clip_trans = transforms.Resize( (224, 224), interpolation=transforms.InterpolationMode.BILINEAR )
    img_model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k') 
    img_adapter = Image_adapter()
    
    # TOKENS
    mask_tokens = [args.mask_tokens]
    placeholder_tokens = [args.placeholder_token1]
    # 1) Add tokens
    tokenizer.add_tokens(mask_tokens)
    tokenizer.add_tokens(placeholder_tokens)
    mask_token_ids = tokenizer.convert_tokens_to_ids(mask_tokens)
    placeholder_token_id1 = tokenizer.convert_tokens_to_ids(placeholder_tokens)
    
    # 2) Resize token_embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    # 3) Load mask/placeholder embeddings
    if args.mask_embed_path is not None:
        mask_embeds=torch.load(args.mask_embed_path)[args.mask_tokens].to(accelerator.device)
        mask_embeds=F.normalize(mask_embeds,p=1,dim=-1)*args.avg_norm
        mask_embeds=mask_embeds.detach()
    

    if args.initialize_token and args.lambda_mlm:
        initializer_token_ids = tokenizer.encode(args.prior_concept1, add_special_tokens=False)
        initializer_token_id = initializer_token_ids[0]
        prior_embed=token_embeds[initializer_token_id].detach().clone().unsqueeze(0)
        for token_id in placeholder_token_id1:
            token_embeds[token_id] = token_embeds[initializer_token_id].clone()
    if args.learned_embed_path1:
        learned_embed1=torch.load(args.learned_embed_path1)#[args.placeholder_token]
        print('load ti embeddings')
        learned_embed1=learned_embed1[args.placeholder_token1]
        with torch.no_grad():
            token_embeds[placeholder_token_id1] = learned_embed1.clone()
        del learned_embed1


    with torch.no_grad():
        learned_embeds_copy=token_embeds[placeholder_token_id1].detach().to(accelerator.device)

    # Context Net
    from contextnet import ContextNet
    if 'stable-diffusion-2-1' in args.pretrained_model_name_or_path:
        cls_net=ContextNet(1024, len(token_embeds)-1)
        cls_output_dim=len(token_embeds)-1
    elif 'stable-diffusion-v1-5' in args.pretrained_model_name_or_path:
        cls_net=ContextNet(768, len(token_embeds))
        cls_output_dim=len(token_embeds)
    else:
        assert False,'undefined sd version'
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # TOKENS
    # We only train the additional adapter LoRA layers
    if vae is not None:
        vae.requires_grad_(False)
    # text_encoder.requires_grad_(False)
    if args.lambda_mlm:
        text_encoder.text_model.encoder.requires_grad_(False)
        text_encoder.text_model.final_layer_norm.requires_grad_(False)
        text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
    else:
        text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    img_model.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    if vae is not None:
        vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # if args.gradient_checkpointing:
    #     unet.enable_gradient_checkpointing()
    #     if args.train_text_encoder:
    #         text_encoder.gradient_checkpointing_enable()

    # now we will add new LoRA weights to the attention layers
    # It's important to realize here how many attention weights will be added and of which sizes
    # The sizes of the attention layers consist only of two different variables:
    # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
    # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

    # Let's first see how many attention processors we will have to set.
    # For Stable Diffusion, it should be equal to:
    # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
    # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
    # - up blocks (2x attention layers) * (3x transformer layers) * (3x up blocks) = 18
    # => 32 layers

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

    # The text encoder comes from 🤗 transformers, so we cannot directly modify it.
    # So, instead, we monkey-patch the forward calls of its attention-blocks.
    # if args.train_text_encoder:
    #     # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
    #     text_lora_parameters = LoraLoaderMixin._modify_text_encoder(text_encoder, dtype=torch.float32, rank=args.rank)

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            unet_lora_layers_to_save = None
            for model in models:
                if isinstance(model, type(accelerator.unwrap_model(unet))):
                    print('unet lora')
                    unet_lora_layers_to_save = unet_lora_state_dict(model)
                elif isinstance(model, type(accelerator.unwrap_model(text_encoder))):
                    print('learned_embeds')
                    learned_embeds=accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[min(placeholder_token_id1) : max(placeholder_token_id1) + 1]
                    learned_embeds_dict = {args.placeholder_token1: learned_embeds.detach().cpu()}
                    torch.save(learned_embeds_dict, os.path.join(output_dir,"learned_embeds.pt"))
                elif isinstance(model, type(accelerator.unwrap_model(cls_net))):
                    print('cls_net lora')
                    # text_encoder_lora_layers_to_save = text_encoder_lora_state_dict(model)
                elif isinstance(model, type(accelerator.unwrap_model(img_adapter))):
                    print('img_adapter')
                    torch.save(model.state_dict() ,os.path.join(output_dir,"adapter.pt"))
                    # text_encoder_lora_layers_to_save = text_encoder_lora_state_dict(model)
                else:
                    print(model)
                    print('else')
                   

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            LoraLoaderMixin.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_ = None

        while len(models) > 0:
            model = models.pop()
            if isinstance(model, type(accelerator.unwrap_model(unet))):
                unet_ = model
            elif isinstance(model, type(accelerator.unwrap_model(text_encoder))):
                text_encoder_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)
        LoraLoaderMixin.load_lora_into_unet(lora_state_dict, network_alphas=network_alphas, unet=unet_)
        LoraLoaderMixin.load_lora_into_text_encoder(
            lora_state_dict, network_alphas=network_alphas, text_encoder=text_encoder_
        )

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

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
    params_to_optimize = (
        [{"params": itertools.chain(unet_lora_parameters), "lr": args.learning_rate},
         {"params": text_encoder.get_input_embeddings().parameters(), "lr": 1e-4},
         {"params": itertools.chain(img_adapter.parameters()), "lr":args.learning_rate_adapter}
        ] if args.lambda_mlm
        else [ {"params": itertools.chain(unet_lora_parameters), "lr": args.learning_rate_adapter},
               {"params": itertools.chain(img_adapter.parameters()), "lr":args.learning_rate_adapter}
            ]
         )
    optimizer = optimizer_class(
        params_to_optimize,
        # lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


    pre_computed_encoder_hidden_states = None
    validation_prompt_encoder_hidden_states = None
    validation_prompt_negative_prompt_embeds = None
    pre_computed_class_prompt_encoder_hidden_states = None

    # Dataset and DataLoaders creation:
    train_dataset = TextualInversionDataset(
        include_prior_concept=args.include_prior_concept,
        data_root=args.train_data_dir1,
        tokenizer=tokenizer,
        size=args.resolution,
        placeholder_token=(" ".join(tokenizer.convert_ids_to_tokens(placeholder_token_id1))),
        repeats=args.repeats,
        center_crop=args.center_crop,
        flip_p=args.flip_p,
        prior_concept=args.prior_concept1,
        mask_token_ids=mask_token_ids[0],
        mlm_target=args.mlm_target,
        get_images=True,
        prompt_type=args.prompt_type,
    )

    train_dataset_mlm = TextualInversionDataset(
        include_prior_concept=args.include_prior_concept,
        data_root=args.train_data_dir1,
        tokenizer=tokenizer,
        size=args.resolution,
        placeholder_token=(" ".join(tokenizer.convert_ids_to_tokens(placeholder_token_id1))),
        repeats=args.repeats,
        center_crop=args.center_crop,
        flip_p=args.flip_p,
        prior_concept=args.prior_concept1,
        mask_token_ids=mask_token_ids[0],
        mlm_target=args.mlm_target,
        get_images=False,
        prompt_type=args.prompt_type,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )
    mlm_loader = torch.utils.data.DataLoader(
            train_dataset_mlm,
            batch_size=args.mlm_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
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
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.lambda_mlm:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler, img_model, img_adapter,cls_net = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler, img_model, img_adapter,cls_net
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler, img_model, img_adapter,cls_net = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler, img_model, img_adapter,cls_net
        )
    if args.cls_net_path is not None:
        for defined_key in cls_net.state_dict():
            print(defined_key,'defined_key-clsnet')
        saved_state_dict = torch.load(args.cls_net_path, map_location=torch.device('cpu'))
        print()
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
        accelerator.init_trackers("dreambooth-lora", config=tracker_config)
        pipeline_args = {}
        # if text_encoder is not None:
        #     pipeline_args["text_encoder"] = unwrap_model(text_encoder)
        # if args.skip_save_text_encoder:
        #     pipeline_args["text_encoder"] = None
        if vae is not None:
            pipeline_args["vae"] = vae
        # pipeline = DiffusionPipeline.from_pretrained(
        #     args.pretrained_model_name_or_path,
        #     unet=unwrap_model(unet),
        #     text_encoder=accelerator.unwrap_model(text_encoder),
        #     revision=args.revision,
        #     variant=args.variant,
        #     **pipeline_args,
        #     feature_extractor=None,
        # safety_checker=None,
        # requires_safety_checker=False,
        # )
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

    # Potentially load in the weights and states from a previous save

    initial_global_step = 0
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    cos_sim=torch.nn.CosineSimilarity().to(accelerator.device)
    orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.lambda_mlm:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # 1. Load Batch
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                is_keyword_tokens = batch["is_keyword_tokens"].to(dtype=weight_dtype)
                input_ids=batch["input_ids"]# B,77 list of booleans (tensor)
                masks=batch["masks"]# B,77 list of booleans (tensor)
                masks64=torch.nn.functional.interpolate(masks,(64,64))
                is_keyword_tokens=batch["is_keyword_tokens"]# B,77 list of booleans (tensor)
                # for MLM
                batch_mlm=load_mlm_batch(mlm_loader)
                is_keyword_tokens_mlm=batch_mlm["is_keyword_tokens_mlm"]
                masked_idxs=batch_mlm["masked_idxs"]
                mlm_labels=batch_mlm["mlm_labels"].to(accelerator.device)
                non_special_idxs=batch_mlm["non_special_idxs"]
                input_ids_masked=batch_mlm["input_ids_masked"].to(accelerator.device)
                input_ids_pos=batch_mlm["input_ids_pos"].to(accelerator.device)

                if vae is not None:
                    # Convert images to latent space
                    model_input = vae.encode(pixel_values).latent_dist.sample()
                    model_input = model_input * vae.config.scaling_factor
                else:
                    model_input = pixel_values

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz, channels, height, width = model_input.shape
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                )
                timesteps = timesteps.long()

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

                # Get the text embedding for conditioning

                # encoder_hidden_states = encode_prompt(
                #         text_encoder,
                #         batch["input_ids"],
                #         batch["attention_mask"],
                #         text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
                #     )
                if args.lambda_mlm:
                    learned_embeds=accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[min(placeholder_token_id1) : max(placeholder_token_id1) + 1]
                    if args.normalize_target1:
                        target_emb=F.normalize(learned_embeds,p=1,dim=-1)*args.normalize_target1
                    else:
                        target_emb=learned_embeds
                    encoder_hidden_states = text_encoder(input_ids,
                                            is_keyword_tokens1=is_keyword_tokens,
                                            inj_embeddings1=target_emb
                                            )[0].to(dtype=weight_dtype)
                else:
                    encoder_hidden_states = text_encoder(input_ids,)[0].to(dtype=weight_dtype)
                with torch.no_grad():
                    img_state = img_model.encode_image( clip_trans(pixel_values) ).unsqueeze(1)
                # Predict the noise residual
                img_state = img_adapter(img_state)
                if accelerator.unwrap_model(unet).config.in_channels == channels * 2:
                    noisy_model_input = torch.cat([noisy_model_input, noisy_model_input], dim=1)
                if args.class_labels_conditioning == "timesteps":
                    class_labels = timesteps
                else:
                    class_labels = None

                
                # SHAPES
                # class_labels: None
                # encoder_hidden_states: 1,77,1024
                # img_state: 1,1,1024
                # MEANING
                # encoder_hidden_states: f_s
                # img_state: f_i


                # Predict the noise residual
                model_pred = unet(noisy_model_input, timesteps, encoder_hidden_states + img_state, class_labels=class_labels).sample
                text_pred = unet(noisy_model_input, timesteps, encoder_hidden_states, class_labels=class_labels).sample

                # if model predicts variance, throw away the prediction. we will only train on the
                # simplified training objective. This means that all schedulers using the fine tuned
                # model must be configured to use one of the fixed variance variance types.
                if model_pred.shape[1] == 6: # NO
                    model_pred, _ = torch.chunk(model_pred, 2, dim=1)



                


                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss_aux1 = F.mse_loss(text_pred.float(), target.float(), reduction="mean")
                loss_aux2 = cal_cos(encoder_hidden_states, img_state, cos)
                loss = loss + (args.lambda_subject*loss_aux1) + (args.lambda_cos*loss_aux2)
                # 3. MLM Loss
                loss_mlm=None
                if args.lambda_mlm:
                    clip_text_embedding_masked = text_encoder(input_ids_masked,
                                                            mask_embedding=mask_embeds.unsqueeze(0),
                                                            mask_idxs=masked_idxs,
                                                            is_keyword_tokens1=is_keyword_tokens_mlm,
                                                            inj_embeddings1=target_emb,
                                                            )[0].to(accelerator.device, dtype=weight_dtype)
                    mlm_logits=cls_net(clip_text_embedding_masked)
                    masked_idxs_flat=masked_idxs.view(-1)
                    loss_mlm = F.cross_entropy(
                        mlm_logits.view(-1,cls_output_dim),
                        mlm_labels.view(-1),
                        ignore_index=-100,
                        reduction='none'
                    )
                    loss_mlm[masked_idxs_flat]*=args.mlm_weight
                    loss_mlm=loss_mlm.mean()
                    loss+=(loss_mlm*args.lambda_mlm)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        unet_lora_parameters
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if args.lambda_mlm:
                    index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
                    assert isinstance(mask_token_ids,list)
                    assert isinstance(placeholder_token_id1,list)
                    # placeholder_token_id1: no_update=False -> update them
                    if args.freeze_mask_embedding:
                        index_no_updates[min(placeholder_token_id1) : max(placeholder_token_id1) + 1] = False
                    else:
                        # no_update = False -> do update them
                        # setting mask_tokens=False -> update mask tokens
                        index_no_updates[min(placeholder_token_id1+mask_token_ids) : max(placeholder_token_id1+mask_token_ids) + 1] = False
                    with torch.no_grad():
                        accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                            index_no_updates
                        ] = orig_embeds_params[index_no_updates]

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if global_step % args.validation_steps == 0:
                    input_image=(pixel_values[0].permute(1,2,0).detach().cpu().numpy()+1)*127.5
                    input_mask=masks[0].permute(1,2,0).detach().cpu().numpy()
                    if args.masked_loss:
                        input_image=input_image*input_mask
                    input_image=input_image.astype(np.uint8)
                    input_image=Image.fromarray(input_image)
                    input_image.save(os.path.join(viz_dir,'input_image_s{:05d}.jpg'.format(global_step)))
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
                    learned_embeds=accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[min(placeholder_token_id1) : max(placeholder_token_id1) + 1]
                    if args.normalize_target1:
                        target_emb=F.normalize(learned_embeds,p=1,dim=-1)*args.normalize_target1
                    else:
                        target_emb=learned_embeds
                    images,validation_prompts = log_validation(
                                    tokenizer=tokenizer, 
                                    args=args, 
                                    accelerator=accelerator, 
                                    target_emb=target_emb.detach(),
                                    pipeline=pipeline,
                                    step=global_step
                                )
                    # save images
                    # validation_files=os.listdir(args.train_data_dir)
                    validation_files=sorted(os.listdir(args.train_data_dir1))
                    validation_target=Image.open(os.path.join((args.train_data_dir1),validation_files[0])).resize((512,512)).convert('RGB')
                    # mod here
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
                        # print(image.size,'image.size')
                        merged_viz=render_caption(merged_viz,val_prompt,[x0,y0+20,x1,y1])
                        merged_viz.paste(image.convert('RGB'),((col_idx+1)*(512+margin_right),row_idx*(512+margin_bottom)))
                    merged_viz.save(os.path.join(sample_dir, 'sample_{:05d}.jpg'.format(global_step)))
                    # mod here
                    torch.cuda.empty_cache()

                if accelerator.is_main_process:
                    if (global_step % args.checkpointing_steps == 0):
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(ckpt_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint-")]
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
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit
                        save_path = os.path.join(ckpt_dir, "checkpoint-{:04d}".format(global_step))
                        os.makedirs(save_path,exist_ok=True)
                        # accelerator.save_state(save_path)
                        save_path_unet=os.path.join(save_path,'unet_{:04d}.pt'.format(global_step))
                        torch.save(unet.state_dict(),save_path_unet)
                        logger.info(f"Saved state to {save_path_unet}")
                        save_path_adapter=os.path.join(save_path,'adapter_{:04d}.pt'.format(global_step))
                        torch.save(img_adapter.state_dict() ,save_path_adapter)
                        logger.info(f"Saved state to {save_path_adapter}")
                        if args.lambda_mlm:
                            learned_embeds=accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[min(placeholder_token_id1) : max(placeholder_token_id1) + 1]
                            save_path_learned_embeds=os.path.join(save_path,'learned_embeds_{:04d}.pt'.format(global_step))
                            learned_embeds_dict = {args.placeholder_token1: learned_embeds.detach().cpu()}
                            torch.save(learned_embeds_dict, save_path_learned_embeds)
                            logger.info(f"Saved state to {save_path_learned_embeds}")
                progress_bar.update(1)
                global_step += 1
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                with torch.no_grad():
                    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[min(placeholder_token_id1) : max(placeholder_token_id1) + 1]
                    if args.normalize_target1:
                        target_emb=F.normalize(learned_embeds,p=1,dim=-1)*args.normalize_target1
                    else:
                        target_emb=learned_embeds
                    norm_target=torch.norm(target_emb,p=1,dim=-1)
                    logs['norm_target']=norm_target.item()

                if loss_mlm is not None:
                    logs['loss_mlm']=loss_mlm.detach().item()
                    orient_dev=cos_sim(learned_embeds_copy,learned_embeds)
                    orient_dev='{:.5f}'.format(orient_dev.item())
                    logs['orient_dev']=orient_dev
                if loss_aux1 is not None:
                    logs['loss_aux1']=loss_aux1.detach().item()
                if loss_aux2 is not None:
                    logs['loss_aux2']=loss_aux2.detach().item()
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                if global_step >= args.max_train_steps:
                    break

    # Save the lora layers
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
