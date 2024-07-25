export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR1="/data/twkim/diffusion/personalization/collected/images/pet_cat1"
export DATA_DIR2="/data/twkim/diffusion/personalization/collected/images/pet_dog1"
export CUDA_VISIBLE_DEVICES=0;
accelerate launch --main_process_port 1234  text_encoder_analysis.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR1 \
  --train_data_dir2=$DATA_DIR2 \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=1001 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="saved_models/analysis" \
  --seed=7776 \
  --mask_tokens="[MASK]" \
  --cls_net_path='saved_models/mlm_contextnet_nonpad_lr1e4/checkpoints/cls_net_99000_ckpt.pt' \
  --mask_embed_path='saved_models/mlm_contextnet_nonpad_lr1e4/checkpoints/mask_embeds_99000_ckpt.pt' \
  --mlm_target='masked' \
  --mlm_batch_size=400 \
  --prompt_type='two_pets' \
  --placeholder_token1="<pet_cat1>" \
  --placeholder_token2="<pet_dog1>" \
  --prior_concept1="cat" \
  --prior_concept2="dog" \
  --resume_unet_path='saved_models/dreambooth_models/multi/tmp_multi_learned_embeds/checkpoints/checkpoint-2000/unet_s2000.pt' \
  --resume_text_encoder_path='saved_models/dreambooth_models/multi/tmp_multi_learned_embeds/checkpoints/checkpoint-2000/text_encoder_s2000.pt' \
  --include_prior_concept=1 \
  --run_name='tmp' \
  --calibrate_kpos=0 \
  --calibrate_kneg=0 \
  --calibrate_pneg=0 \
  --calibrate_ppos=0 


  