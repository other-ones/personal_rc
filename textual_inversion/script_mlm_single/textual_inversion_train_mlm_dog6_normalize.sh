

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/dog6"
export CUDA_VISIBLE_DEVICES=0;
accelerate launch --main_process_port 4234  textual_inversion_train_mlm_single.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token1="<dog6>" \
  --prior_concept1="dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3001 \
  --learning_rate=5e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="saved_models/single/dog6" \
  --seed=7777 \
  --mask_tokens="[MASK]" \
  --lambda_mlm=0.0005 --freeze_mask_embedding=1 \
  --cls_net_path='saved_models/mlm_contextnet_nonpad_lr1e4/checkpoints/cls_net_99000_ckpt.pt' \
  --mask_embed_path='saved_models/mlm_contextnet_nonpad_lr1e4/checkpoints/mask_embeds_99000_ckpt.pt' \
  --mlm_target='masked' \
  --mlm_batch_size=50 \
  --run_name='ti_norm8_noprior_mlm00005_dog6' \
  --scale_lr \
  --prompt_type='pet' \
  --normalize_target=8 \
  --include_prior_concept=0 



export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/dog6"
export CUDA_VISIBLE_DEVICES=1;
accelerate launch --main_process_port 4235  textual_inversion_train_mlm_single.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token1="<dog6>" \
  --prior_concept1="dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3001 \
  --learning_rate=5e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="saved_models/single/dog6" \
  --seed=7777 \
  --mask_tokens="[MASK]" \
  --lambda_mlm=0.001 --freeze_mask_embedding=1 \
  --cls_net_path='saved_models/mlm_contextnet_nonpad_lr1e4/checkpoints/cls_net_99000_ckpt.pt' \
  --mask_embed_path='saved_models/mlm_contextnet_nonpad_lr1e4/checkpoints/mask_embeds_99000_ckpt.pt' \
  --mlm_target='masked' \
  --mlm_batch_size=50 \
  --run_name='ti_norm8_noprior_mlm0001_dog6' \
  --scale_lr \
  --prompt_type='pet' \
  --normalize_target=8 \
  --include_prior_concept=0 

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/dog6"
export CUDA_VISIBLE_DEVICES=2;
accelerate launch --main_process_port 4236  textual_inversion_train_mlm_single.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token1="<dog6>" \
  --prior_concept1="dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3001 \
  --learning_rate=5e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="saved_models/single/dog6" \
  --seed=7777 \
  --mask_tokens="[MASK]" \
  --lambda_mlm=0.0025 --freeze_mask_embedding=1 \
  --cls_net_path='saved_models/mlm_contextnet_nonpad_lr1e4/checkpoints/cls_net_99000_ckpt.pt' \
  --mask_embed_path='saved_models/mlm_contextnet_nonpad_lr1e4/checkpoints/mask_embeds_99000_ckpt.pt' \
  --mlm_target='masked' \
  --mlm_batch_size=50 \
  --run_name='ti_norm8_noprior_mlm00025_dog6' \
  --scale_lr \
  --prompt_type='pet' \
  --normalize_target=8 \
  --include_prior_concept=0 




export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/dog6"
export CUDA_VISIBLE_DEVICES=3;
accelerate launch --main_process_port 4236  textual_inversion_train_mlm_single.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token1="<dog6>" \
  --prior_concept1="dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3001 \
  --learning_rate=5e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="saved_models/single/dog6" \
  --seed=7777 \
  --mask_tokens="[MASK]" \
  --lambda_mlm=0 --freeze_mask_embedding=1 \
  --cls_net_path='saved_models/mlm_contextnet_nonpad_lr1e4/checkpoints/cls_net_99000_ckpt.pt' \
  --mask_embed_path='saved_models/mlm_contextnet_nonpad_lr1e4/checkpoints/mask_embeds_99000_ckpt.pt' \
  --mlm_target='masked' \
  --mlm_batch_size=50 \
  --run_name='ti_norm8_noprior_no_mlm_dog6' \
  --scale_lr \
  --prompt_type='pet' \
  --normalize_target=8 \
  --include_prior_concept=0 




