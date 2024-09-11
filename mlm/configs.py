import os
import argparse
from lora_diffusion import (
    safetensors_available,
    save_lora_weight,
    save_safeloras_with_embeds,
)
def parse_args(input_args=None):
    
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # Added
    parser.add_argument("--normalize_mask_embeds",type=int,default=0)
    parser.add_argument("--eval_batch_size",type=int,default=4,help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--nsamples",type=int,default=4,help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--real_caption_path",type=str)
    parser.add_argument("--synth_caption_root",type=str)
    parser.add_argument("--trained_keys",type=str)
    parser.add_argument("--lambda_reg",type=float,default=0,help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--lambda_sparsity",type=float,default=0,help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--lambda_text",type=float,default=0,help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--lambda_momentum",type=float,default=0,help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--lambda_sim",type=float,default=0,help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--validation_image_path",type=str,help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--tracked_token",type=str,help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--init_scale",type=float,default=0.01)
    parser.add_argument("--momentum_warmup",type=int,default=100)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--resume_mapper_path',type=str)
    parser.add_argument('--project_name',type=str)
    parser.add_argument('--scale_personal',type=float,default=0)
    parser.add_argument('--scale_general',type=float,default=0)
    parser.add_argument('--checkpoints_total_limit', default=10,type=int)
    parser.add_argument('--fixed_target',type=str)
    parser.add_argument('--mapper_personal_path',type=str)
    parser.add_argument('--mapper_general_path',type=str)
    parser.add_argument('--val_steps',type=int)
    parser.add_argument('--fix_momentum',action='store_true')
    parser.add_argument('--mask_tokens',type=str)
    parser.add_argument('--caption_path',type=str)
    parser.add_argument('--mlm_weight',type=float,default=1)
    parser.add_argument('--mlm_target',type=str)
    parser.add_argument('--whole_word_mask',type=int,default=0)
    parser.add_argument('--num_vectors',type=int)
    parser.add_argument("--lambda_target",type=float,default=1,help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--avg_norm",type=float,default=8.525119902624288)
    parser.add_argument("--lambda_mlm",type=float,default=1)
    parser.add_argument("--cls_net_path",type=str)
    parser.add_argument("--mask_prob",type=float,default=0.25)
    parser.add_argument("--mask_embed_path",type=str)
    parser.add_argument("--log_mlm_steps",type=int,default=25)
    parser.add_argument("--arcface_model_name",type=str,default='r100')
    parser.add_argument("--arcface_model_path",type=str,default='arcface_torch/weights/ms1mv3_arcface_r100_fp16/backbone.pth')
    parser.add_argument("--mapper_path",type=str)
    parser.add_argument("--scale_embeddings",type=float)


    # Token placeholder
    parser.add_argument("--momentum_decay",type=float,default=0.99)
    # parser.add_argument("--general_token",type=str)
    # parser.add_argument("--personal_token",type=str)
    # parser.add_argument("--initializer_general",type=str)
    # parser.add_argument("--initializer_personal",type=str)
    # parser.add_argument('--num_vectors_general', default=1,type=int)
    # parser.add_argument("--num_vectors_personal",type=int,default=5,help="Batch size (per device) for the evaluation dataloader.")


    # Added
    
    parser.add_argument('--validation_prompt', type=str)
    parser.add_argument('--report_to', type=str)
    parser.add_argument('--checkpointing_steps', default=100,type=int)
    parser.add_argument('--mask_bg', default=1,type=float)
    parser.add_argument('--mask_fg', default=1,type=float)
    parser.add_argument('--resume_from_checkpoint', type=str)
    parser.add_argument('--random_font', default=False,action='store_true')
    parser.add_argument('--fixed_layout', default=False,action='store_true')
    parser.add_argument('--run_name', type=str)
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument('--guidance_strength', default=1,type=float)
    parser.add_argument('--num_intervention_steps', default=20,type=int)
    parser.add_argument('--num_inference_steps', default=100,type=int)
    parser.add_argument('--guidance_scale', default=7.5,type=float)
    parser.add_argument('--roi_weight', default=1,type=float)
    parser.add_argument('--bg_weight', default=1,type=float)
    parser.add_argument('--learned_embed_path', type=str)
    parser.add_argument('--uniform_layout', action='store_true')
    parser.add_argument('--dbname', type=str)
    parser.add_argument('--target_word', type=str)
    parser.add_argument('--resume_unet_ckpt_path', type=str)
    parser.add_argument('--resume_recnet_ckpt_path', type=str)
    parser.add_argument('--rec_loss_weight',type=float)
    parser.add_argument('--exclude_suffix', action='store_true')
    parser.add_argument('--max_char_seq_length', default=26,type=float)
    parser.add_argument('--max_tstep', default=200,type=int)
    parser.add_argument('--dataloader_num_workers', default=0,type=int)
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained vae or vae identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_root",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    
    parser.add_argument(
        "--initializer_token",
        type=str,
        default="<>",
        required=False,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If not have enough images, additional images will be"
            " sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["pt", "safe", "both"],
        default="both",
        help="The output format of the model predicitions and checkpoints.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop images before resizing to resolution",
    )
    parser.add_argument(
        "--color_jitter",
        action="store_true",
        help="Whether to apply color jitter to images",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument('--visualize_steps',type=int,default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help="Save checkpoint every X updates steps.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="Rank of LoRA approximation.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_text",
        type=float,
        default=5e-6,
        help="Initial learning rate for text encoder (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_ti",
        type=float,
        default=5e-4,
        help="Initial learning rate for embedding of textual inversion (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=0, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--unfreeze_lora_step",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--just_ti",
        action="store_true",
        help="Debug to see just ti",
    )
    parser.add_argument(
        "--resize",
        type=bool,
        default=True,
        required=False,
        help="Should images be resized to --resolution before training?",
    )
    parser.add_argument(
        "--use_xformers", action="store_true", help="Whether or not to use xformers"
    )
    parser.add_argument(
        "--instance_data_list")
    parser.add_argument('--charset',default='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~')
    parser.add_argument('--max_length',default=13)

    parser.add_argument(
        "--root_dir",
        type=str,
        default='/data/dataset',
        required=False,
        help="A folder containing the training data of instance images.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        if args.class_data_dir is not None:
            logger.warning(
                "You need not use --class_data_dir without --with_prior_preservation."
            )
        if args.class_prompt is not None:
            logger.warning(
                "You need not use --class_prompt without --with_prior_preservation."
            )

    if not safetensors_available:
        if args.output_format == "both":
            print(
                "Safetensors is not available - changing output format to just output PyTorch files"
            )
            args.output_format = "pt"
        elif args.output_format == "safe":
            raise ValueError(
                "Safetensors is not available - either install it, or change output_format."
            )

    return args
