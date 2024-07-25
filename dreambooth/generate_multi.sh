
export DATA_DIR1="/data/twkim/diffusion/personalization/collected/images/pet_cat1"
export DATA_DIR2="/data/twkim/diffusion/personalization/collected/images/pet_dog1"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=0;
accelerate launch --main_process_port 2730  generate_multi.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR1 \
  --train_data_dir2=$DATA_DIR2 \
  --resolution=512 \
  --output_dir="results/multi/calibrate/pet_cat1_pet_dog1" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --resume_unet_path='saved_models/dreambooth_models/multi/tmp_multi_learned_embeds_sim01_margin0/checkpoints/checkpoint-2000/unet_s2000.pt' \
  --resume_text_encoder_path='saved_models/dreambooth_models/multi/tmp_multi_learned_embeds_sim01_margin0/checkpoints/checkpoint-2000/text_encoder_s2000.pt' \
  --placeholder_token1="<pet_cat1>" \
  --prior_concept1='cat' \
  --prior_concept2='dog' \
  --placeholder_token2="<pet_dog1>" \
  --include_prior_concept=1 \
  --prompt_type='two_pets' \
  --eval_batch_size=5 \
  --num_images_per_prompt=15 \
  --calibrate_kpos=0 \
  --calibrate_ppos=1 \
  --calibrate_kneg=0 \
  --calibrate_pneg=1 \
  --break_num=3




export DATA_DIR1="/data/twkim/diffusion/personalization/collected/images/pet_cat1"
export DATA_DIR2="/data/twkim/diffusion/personalization/collected/images/pet_dog1"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=1;
accelerate launch --main_process_port 2731  generate_multi.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR1 \
  --train_data_dir2=$DATA_DIR2 \
  --resolution=512 \
  --output_dir="results/multi/calibrate/pet_cat1_pet_dog1" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --resume_unet_path='saved_models/dreambooth_models/multi/tmp_multi_learned_embeds_sim01_margin0/checkpoints/checkpoint-2000/unet_s2000.pt' \
  --resume_text_encoder_path='saved_models/dreambooth_models/multi/tmp_multi_learned_embeds_sim01_margin0/checkpoints/checkpoint-2000/text_encoder_s2000.pt' \
  --placeholder_token1="<pet_cat1>" \
  --prior_concept1='cat' \
  --prior_concept2='dog' \
  --placeholder_token2="<pet_dog1>" \
  --include_prior_concept=1 \
  --prompt_type='two_pets' \
  --eval_batch_size=5 \
  --num_images_per_prompt=15 \
  --calibrate_kpos=0 \
  --calibrate_ppos=10 \
  --calibrate_kneg=0 \
  --calibrate_pneg=10 \
  --break_num=3

