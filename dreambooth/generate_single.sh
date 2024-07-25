export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/pet_cat1"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=1;
accelerate launch --main_process_port 2731  generate_single.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<pet_cat1>" \
  --resolution=512 \
  --output_dir="results/single/pet_cat1" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --resume_unet_path='saved_models/single/pet_cat1/dreambooth_pp_nomlm_pet_cat1/checkpoints/checkpoint-1000/unet_s1000.pt' \
  --resume_text_encoder_path='saved_models/single/pet_cat1/dreambooth_pp_nomlm_pet_cat1/checkpoints/checkpoint-1000/text_encoder_s1000.pt' \
  --prior_concept1='cat' \
  --include_prior_concept=1 \
  --prompt_type='pet' \
  --eval_batch_size=20 \
  --num_images_per_prompt=15


export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/pet_cat1"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=0;
accelerate launch --main_process_port 2730  generate_single.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<pet_cat1>" \
  --resolution=512 \
  --output_dir="results/single/pet_cat1" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --resume_unet_path='saved_models/single/pet_cat1/dreambooth_pp_mlm01_pet_cat1/checkpoints/checkpoint-1000/unet_s1000.pt' \
  --resume_text_encoder_path='saved_models/single/pet_cat1/dreambooth_pp_mlm01_pet_cat1/checkpoints/checkpoint-1000/text_encoder_s1000.pt' \
  --prior_concept1='cat' \
  --include_prior_concept=1 \
  --prompt_type='pet' \
  --eval_batch_size=20 \
  --num_images_per_prompt=15
