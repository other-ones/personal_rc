export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/pet_dog1"
export CUDA_VISIBLE_DEVICES=1;
accelerate launch --main_process_port 4230  train_mlm_single.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token1="<pet_dog1>" \
  --prior_concept1="dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3001 \
  --learning_rate=5e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="saved_models/tmp" \
  --seed=7777 \
  --mask_tokens="[MASK]" \
  --lambda_mlm=0.001 --freeze_mask_embedding=1 \
  --cls_net_path='saved_models/mlm_models/sd1_contextnetv4_nonpadding_1e4_unnorm_mprob015_batch150/checkpoints/checkpoint-60000/cls_net_60000_ckpt.pt' \
  --mask_embed_path='saved_models/mlm_models/sd1_contextnetv4_nonpadding_1e4_unnorm_mprob015_batch150/checkpoints/checkpoint-60000/mask_embeds_60000_ckpt.pt' \
  --mlm_target='masked' \
  --mlm_batch_size=50 \
  --run_name='tmp_mlm0001' \
  --scale_lr \
  --train_prompt_type='pet' \
  --eval_prompt_type='cat' \
  --include_prior_concept=1 \
  --rev=1 \
  --caption_root='../datasets_pkgs/captions/v5_simple'


export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/pet_dog1"
export CUDA_VISIBLE_DEVICES=2;
accelerate launch --main_process_port 4230  train_mlm_single.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token1="<pet_dog1>" \
  --prior_concept1="dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3001 \
  --learning_rate=5e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="saved_models/tmp" \
  --seed=7777 \
  --mask_tokens="[MASK]" \
  --lambda_mlm=0 --freeze_mask_embedding=1 \
  --cls_net_path='saved_models/mlm_models/sd1_contextnetv4_nonpadding_1e4_unnorm_mprob015_batch150/checkpoints/checkpoint-60000/cls_net_60000_ckpt.pt' \
  --mask_embed_path='saved_models/mlm_models/sd1_contextnetv4_nonpadding_1e4_unnorm_mprob015_batch150/checkpoints/checkpoint-60000/mask_embeds_60000_ckpt.pt' \
  --mlm_target='masked' \
  --mlm_batch_size=50 \
  --run_name='tmp_nomlm' \
  --scale_lr \
  --train_prompt_type='pet' \
  --eval_prompt_type='cat' \
  --include_prior_concept=1 \
  --rev=1 \
  --caption_root='../datasets_pkgs/captions/v5_simple'


command+='--cls_net_path="saved_models/mlm_models/sd1_contextnetv4_nonpadding_1e4_unnorm_mprob015_batch150/checkpoints/checkpoint-60000/cls_net_60000_ckpt.pt" \\\n'
command+='--mask_embed_path="saved_models/mlm_models/sd1_contextnetv4_nonpadding_1e4_unnorm_mprob015_batch150/checkpoints/checkpoint-60000/mask_embeds_60000_ckpt.pt" \\\n'
