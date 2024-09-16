export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/pet_dog1"
export CUBLAS_WORKSPACE_CONFIG=:4096:8;
export CUDA_VISIBLE_DEVICES=4;
accelerate launch --main_process_port 1234  pplus_train_clean.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<pet_dog1>" \
  --prior_concept1="dog" \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=3001 \
  --learning_rate=5e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="saved_models/pplus_models/" \
  --seed=7777 \
  --mask_tokens="[MASK]" \
  --lambda_mlm=0.001 --freeze_mask_embedding=1 \
  --cls_net_path='saved_models/mlm_models/sd1_contextnetv6_nonpadding_1e4_unnorm_mprob015_batch150_bigger_synthcap/checkpoints/checkpoint-100000/cls_net_100000_ckpt.pt' \
  --mask_embed_path='saved_models/mlm_models/sd1_contextnetv6_nonpadding_1e4_unnorm_mprob015_batch150_bigger_synthcap/checkpoints/checkpoint-100000/mask_embeds_100000_ckpt.pt' \
  --mlm_target='masked' \
  --mlm_batch_size=10 \
  --run_name='tmp_pplus_mlm_lr5e4' \
  --scale_lr \
  --train_prompt_type='pet' \
  --eval_prompt_type='living' \
  --initializer_token='dog' \
  --include_prior_concept=0 \
  --num_vectors1=7 \
  --mask_prob=0.15 \
  --validation_steps=2 \
  --debug \
  --caption_root='../datasets_pkgs/captions/v7'



export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/pet_dog1"
export CUBLAS_WORKSPACE_CONFIG=:4096:8;
export CUDA_VISIBLE_DEVICES=7;
accelerate launch --main_process_port 7777  pplus_train_clean.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<pet_dog1>" \
  --prior_concept1="dog" \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=3001 \
  --learning_rate=5e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="saved_models/pplus_models/" \
  --seed=7777 \
  --mask_tokens="[MASK]" \
  --lambda_mlm=0.0001 --freeze_mask_embedding=1 \
  --cls_net_path='saved_models/mlm_models/sd1_contextnetv6_nonpadding_1e4_unnorm_mprob015_batch150_bigger_synthcap/checkpoints/checkpoint-100000/cls_net_100000_ckpt.pt' \
  --mask_embed_path='saved_models/mlm_models/sd1_contextnetv6_nonpadding_1e4_unnorm_mprob015_batch150_bigger_synthcap/checkpoints/checkpoint-100000/mask_embeds_100000_ckpt.pt' \
  --mlm_target='masked' \
  --mlm_batch_size=10 \
  --run_name='tmp_mlm' \
  --scale_lr \
  --train_prompt_type='pet' \
  --eval_prompt_type='living' \
  --initializer_token='dog' \
  --include_prior_concept=0 \
  --num_vectors1=7 \
  --mask_prob=0.15 \
  --mlm_idxs=2,3,4 \
  --caption_root='../datasets_pkgs/captions/v7'
# 0,1,2,3,4,5,6


export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/pet_dog1"
export CUBLAS_WORKSPACE_CONFIG=:4096:8;
export CUDA_VISIBLE_DEVICES=6;
accelerate launch --main_process_port 4232  pplus_train_clean.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<pet_dog1>" \
  --prior_concept1="dog" \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=3001 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="saved_models/pplus_models/" \
  --seed=7777 \
  --mask_tokens="[MASK]" \
  --lambda_mlm=0 --freeze_mask_embedding=1 \
  --cls_net_path='saved_models/mlm_models/sd1_contextnetv6_nonpadding_1e4_unnorm_mprob015_batch150_bigger_synthcap/checkpoints/checkpoint-100000/cls_net_100000_ckpt.pt' \
  --mask_embed_path='saved_models/mlm_models/sd1_contextnetv6_nonpadding_1e4_unnorm_mprob015_batch150_bigger_synthcap/checkpoints/checkpoint-100000/mask_embeds_100000_ckpt.pt' \
  --mlm_target='masked' \
  --mlm_batch_size=0 \
  --run_name='tmp_pplus_nomlm_lr1e4' \
  --scale_lr \
  --train_prompt_type='pet' \
  --eval_prompt_type='living' \
  --initializer_token='dog' \
  --include_prior_concept=0 \
  --num_vectors1=7 \
  --mask_prob=0.15 \
  --caption_root='../datasets_pkgs/captions/v7'
