export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUBLAS_WORKSPACE_CONFIG=:4096:8;
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/chair1";
export CUDA_VISIBLE_DEVICES=6;
accelerate launch --main_process_port 4236  ti_train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token1="<chair1>" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3001 \
  --learning_rate=5e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="saved_models/ti_models/disen/chair1" \
  --seed=7777 \
  --mask_tokens="[MASK]" \
  --lambda_mlm=0 --freeze_mask_embedding=1 \
  --cls_net_path='saved_models/mlm_models/sd1_contextnetv4_nonpadding_1e4_unnorm_mprob015_batch150/checkpoints/checkpoint-100000/cls_net_100000_ckpt.pt' \
  --mask_embed_path='saved_models/mlm_models/sd1_contextnetv4_nonpadding_1e4_unnorm_mprob015_batch150/checkpoints/checkpoint-100000/mask_embeds_100000_ckpt.pt' \
  --mlm_target='masked' \
  --mlm_batch_size=25 \
  --run_name='tmp_prior_single_chair1_nomlm_pot' \
  --scale_lr \
  --train_prompt_type='pots' \
  --eval_prompt_type='nonliving' \
  --include_prior_concept=1 \
  --caption_root='../datasets_pkgs/captions/specific' \
  --target_image='kara-eads-xRyL63AwZFE-unsplash.jpg' \
  --initializer_token='pot' \
  --train_prior_concept1='pot'

export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUBLAS_WORKSPACE_CONFIG=:4096:8;
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/chair1";
export CUDA_VISIBLE_DEVICES=7;
accelerate launch --main_process_port 4236  ti_train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token1="<chair1>" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3001 \
  --learning_rate=5e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="saved_models/ti_models/disen/chair1" \
  --seed=7777 \
  --mask_tokens="[MASK]" \
  --lambda_mlm=0.001 --freeze_mask_embedding=1 \
  --cls_net_path='saved_models/mlm_models/sd1_contextnetv4_nonpadding_1e4_unnorm_mprob015_batch150/checkpoints/checkpoint-100000/cls_net_100000_ckpt.pt' \
  --mask_embed_path='saved_models/mlm_models/sd1_contextnetv4_nonpadding_1e4_unnorm_mprob015_batch150/checkpoints/checkpoint-100000/mask_embeds_100000_ckpt.pt' \
  --mlm_target='masked' \
  --mlm_batch_size=25 \
  --run_name='tmp_prior_single_chair1_mlm0001_pot' \
  --scale_lr \
  --train_prompt_type='pots' \
  --eval_prompt_type='nonliving' \
  --include_prior_concept=1 \
  --caption_root='../datasets_pkgs/captions/specific' \
  --target_image='kara-eads-xRyL63AwZFE-unsplash.jpg' \
  --train_prior_concept1='pot' \
  --initializer_token='pot' 

export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUBLAS_WORKSPACE_CONFIG=:4096:8;
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/chair1";
export CUDA_VISIBLE_DEVICES=2;
accelerate launch --main_process_port 4232  ti_train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token1="<chair1>" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3001 \
  --learning_rate=5e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="saved_models/ti_models/disen/chair1" \
  --seed=7777 \
  --mask_tokens="[MASK]" \
  --lambda_mlm=0.001 --freeze_mask_embedding=1 \
  --cls_net_path='saved_models/mlm_models/sd1_contextnetv4_nonpadding_1e4_unnorm_mprob015_batch150/checkpoints/checkpoint-100000/cls_net_100000_ckpt.pt' \
  --mask_embed_path='saved_models/mlm_models/sd1_contextnetv4_nonpadding_1e4_unnorm_mprob015_batch150/checkpoints/checkpoint-100000/mask_embeds_100000_ckpt.pt' \
  --mlm_target='masked' \
  --mlm_batch_size=25 \
  --run_name='tmp_no_prior_single_chair1_mlm0001_chair_specific' \
  --scale_lr \
  --train_prompt_type='chairs' \
  --eval_prompt_type='nonliving' \
  --include_prior_concept=0 \
  --caption_root='../datasets_pkgs/captions/specific' \
  --target_image='scott-webb-eD853mTbBA0-unsplash.jpg' \
  --train_prior_concept1='chair' \
  --initializer_token='chair' 

export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUBLAS_WORKSPACE_CONFIG=:4096:8;
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/chair1";
export CUDA_VISIBLE_DEVICES=7;
accelerate launch --main_process_port 4237  ti_train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token1="<chair1>" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3001 \
  --learning_rate=5e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="saved_models/ti_models/disen/chair1" \
  --seed=7777 \
  --mask_tokens="[MASK]" \
  --lambda_mlm=0 --freeze_mask_embedding=1 \
  --cls_net_path='saved_models/mlm_models/sd1_contextnetv4_nonpadding_1e4_unnorm_mprob015_batch150/checkpoints/checkpoint-100000/cls_net_100000_ckpt.pt' \
  --mask_embed_path='saved_models/mlm_models/sd1_contextnetv4_nonpadding_1e4_unnorm_mprob015_batch150/checkpoints/checkpoint-100000/mask_embeds_100000_ckpt.pt' \
  --mlm_target='masked' \
  --mlm_batch_size=25 \
  --run_name='tmp_prior_single_chair1_nomlm' \
  --scale_lr \
  --train_prompt_type='chairs' \
  --eval_prompt_type='nonliving' \
  --include_prior_concept=1 \
  --caption_root='../datasets_pkgs/captions/specific' \
  --target_image='scott-webb-eD853mTbBA0-unsplash.jpg' \
  --train_prior_concept1='chair' \
  --initializer_token='chair' 