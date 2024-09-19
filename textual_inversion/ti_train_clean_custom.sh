export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export DATA_DIR="/data/twkim/diffusion/personalization/custom_images/charmander/simple";
export CUBLAS_WORKSPACE_CONFIG=:4096:8;
export CUDA_VISIBLE_DEVICES=0;
accelerate launch --main_process_port 4230  ti_train_clean.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token1="<charmander>" \
  --train_prior_concept1="toy" \
  --eval_prior_concept1="stuffed animal " \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=3001 \
  --learning_rate=5e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="saved_models/ti_models/custom/charmander" \
  --seed=7777 \
  --mask_tokens="[MASK]" \
  --lambda_mlm=0.001 --freeze_mask_embedding=1 \
  --cls_net_path='saved_models/mlm_models/sd1_contextnetv6_nonpadding_1e4_unnorm_mprob015_batch150_bigger_synthcap/checkpoints/checkpoint-100000/cls_net_100000_ckpt.pt' \
  --mask_embed_path='saved_models/mlm_models/sd1_contextnetv6_nonpadding_1e4_unnorm_mprob015_batch150_bigger_synthcap/checkpoints/checkpoint-100000/mask_embeds_100000_ckpt.pt' \
  --mlm_target='masked' \
  --mlm_batch_size=25 \
  --run_name='charmander_mlm0001_living' \
  --scale_lr \
  --train_prompt_type='pet' \
  --eval_prompt_type='living' \
  --include_prior_concept=1 \
  --initializer_token='toy' \
  --validation_steps=100 \
  --caption_root='../datasets_pkgs/captions/v7'


export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export DATA_DIR="/data/twkim/diffusion/personalization/custom_images/charmander/simple";
export CUBLAS_WORKSPACE_CONFIG=:4096:8;
export CUDA_VISIBLE_DEVICES=1;
accelerate launch --main_process_port 4231  ti_train_clean.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token1="<charmander>" \
  --train_prior_concept1="toy" \
  --eval_prior_concept1="stuffed animal " \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=3001 \
  --learning_rate=5e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="saved_models/ti_models/custom/charmander" \
  --seed=7777 \
  --mask_tokens="[MASK]" \
  --lambda_mlm=0 --freeze_mask_embedding=1 \
  --cls_net_path='saved_models/mlm_models/sd1_contextnetv6_nonpadding_1e4_unnorm_mprob015_batch150_bigger_synthcap/checkpoints/checkpoint-100000/cls_net_100000_ckpt.pt' \
  --mask_embed_path='saved_models/mlm_models/sd1_contextnetv6_nonpadding_1e4_unnorm_mprob015_batch150_bigger_synthcap/checkpoints/checkpoint-100000/mask_embeds_100000_ckpt.pt' \
  --mlm_target='masked' \
  --mlm_batch_size=25 \
  --run_name='charmander_nomlm' \
  --scale_lr \
  --train_prompt_type='pet' \
  --eval_prompt_type='living' \
  --include_prior_concept=1 \
  --initializer_token='toy' \
  --validation_steps=100 \
  --caption_root='../datasets_pkgs/captions/v7'



export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export DATA_DIR="/data/twkim/diffusion/personalization/custom_images/charmander/simple";
export CUBLAS_WORKSPACE_CONFIG=:4096:8;
export CUDA_VISIBLE_DEVICES=7;
accelerate launch --main_process_port 4237  ti_train_clean.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token1="<charmander>" \
  --train_prior_concept1="toy" \
  --eval_prior_concept1="stuffed animal " \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=3001 \
  --learning_rate=5e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="saved_models/ti_models/custom/charmander" \
  --seed=7777 \
  --mask_tokens="[MASK]" \
  --lambda_mlm=0.001 --freeze_mask_embedding=1 \
  --cls_net_path='saved_models/mlm_models/sd1_contextnetv6_nonpadding_1e4_unnorm_mprob015_batch150_bigger_synthcap/checkpoints/checkpoint-100000/cls_net_100000_ckpt.pt' \
  --mask_embed_path='saved_models/mlm_models/sd1_contextnetv6_nonpadding_1e4_unnorm_mprob015_batch150_bigger_synthcap/checkpoints/checkpoint-100000/mask_embeds_100000_ckpt.pt' \
  --mlm_target='masked' \
  --mlm_batch_size=25 \
  --run_name='charmander_mlm0001_nonliving' \
  --scale_lr \
  --train_prompt_type='nonliving' \
  --eval_prompt_type='nonliving' \
  --include_prior_concept=1 \
  --initializer_token='toy' \
  --validation_steps=100 \
  --caption_root='../datasets_pkgs/captions/v7'
