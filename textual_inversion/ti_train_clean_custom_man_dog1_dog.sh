export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export DATA_DIR="/home/twkim/project/rich_context/textual_inversion/custom_images/man_dog1";
export CUBLAS_WORKSPACE_CONFIG=:4096:8;
export CUDA_VISIBLE_DEVICES=2;
accelerate launch --main_process_port 4230  ti_train_clean.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token1="<man_dog1_dog>" \
  --train_prior_concept1="dog" \
  --eval_prior_concept1="dog " \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=3001 \
  --learning_rate=5e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="saved_models/ti_models/custom/man_dog1_dog" \
  --seed=7777 \
  --mask_tokens="[MASK]" \
  --lambda_mlm=0.0001 --freeze_mask_embedding=1 \
  --cls_net_path='saved_models/mlm_models/sd1_contextnetv6_nonpadding_1e4_unnorm_mprob015_batch150_bigger_synthcap/checkpoints/checkpoint-100000/cls_net_100000_ckpt.pt' \
  --mask_embed_path='saved_models/mlm_models/sd1_contextnetv6_nonpadding_1e4_unnorm_mprob015_batch150_bigger_synthcap/checkpoints/checkpoint-100000/mask_embeds_100000_ckpt.pt' \
  --mlm_target='masked' \
  --mlm_batch_size=25 \
  --run_name='man_dog1_dog_mlm0001_living_re' \
  --scale_lr \
  --train_prompt_type='dogs' \
  --eval_prompt_type='living' \
  --include_prior_concept=1 \
  --initializer_token='dog' \
  --validation_steps=100 \
  --caption_root='../datasets_pkgs/captions/specific'

export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export DATA_DIR="/home/twkim/project/rich_context/textual_inversion/custom_images/man_dog1";
export CUBLAS_WORKSPACE_CONFIG=:4096:8;
export CUDA_VISIBLE_DEVICES=4;
accelerate launch --main_process_port 4231  ti_train_clean.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token1="<man_dog1_dog>" \
  --train_prior_concept1="dog" \
  --eval_prior_concept1="dog " \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=3001 \
  --learning_rate=5e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="saved_models/ti_models/custom/man_dog1_dog" \
  --seed=7777 \
  --mask_tokens="[MASK]" \
  --lambda_mlm=0 --freeze_mask_embedding=1 \
  --cls_net_path='saved_models/mlm_models/sd1_contextnetv6_nonpadding_1e4_unnorm_mprob015_batch150_bigger_synthcap/checkpoints/checkpoint-100000/cls_net_100000_ckpt.pt' \
  --mask_embed_path='saved_models/mlm_models/sd1_contextnetv6_nonpadding_1e4_unnorm_mprob015_batch150_bigger_synthcap/checkpoints/checkpoint-100000/mask_embeds_100000_ckpt.pt' \
  --mlm_target='masked' \
  --mlm_batch_size=25 \
  --run_name='man_dog1_dog_nomlm_living_re' \
  --scale_lr \
  --train_prompt_type='dogs' \
  --eval_prompt_type='living' \
  --include_prior_concept=1 \
  --initializer_token='dog' \
  --validation_steps=100 \
  --caption_root='../datasets_pkgs/captions/specific'


