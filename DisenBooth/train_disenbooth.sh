export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/dog6";
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base";
export OUTPUT_DIR="./saved_models/disenbooth_models";
export CUDA_VISIBLE_DEVICES=7;
export PYTHONPAHT=$PWD;
accelerate launch train_disenbooth_single.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_data_dir1=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --placeholder_token1="<dog6>" \
  --prior_concept1="dog" \
  --include_prior_concept=1 \
  --mask_tokens="[MASK]" \
  --mlm_target='masked' \
  --cls_net_path='saved_models/mlm_models/sd2_contextnet_nonpadding_1e4/checkpoints/cls_net_100000_ckpt.pt' \
  --mask_embed_path='saved_models/mlm_models/sd2_contextnet_nonpadding_1e4/checkpoints/mask_embeds_100000_ckpt.pt' \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=200 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_prompt="A <dog6> dog in a bucket" \
  --validation_steps=200 \
  --seed="0" \
  --run_name='tmp_mlm0001_dog6' \
  --lambda_mlm=0.001 \
  --prompt_type='pet' \
  --train_text_encoder
