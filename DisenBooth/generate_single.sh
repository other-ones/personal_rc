export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/pet_dog1";
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base";
export CUDA_VISIBLE_DEVICES=0;
accelerate launch --main_process_port 2731  generate_single.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<pet_dog1>" \
  --resolution=512 \
  --output_dir="results/disenbooth/sd2/single/pet_dog1" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --resume_lora_path='saved_models/disenbooth_models/sd2/single/pet_dog1/disenbooth_nopp_mlm0001_pet_dog1_train_text/checkpoints/checkpoint-2000' \
  --prior_concept1='dog' \
  --include_prior_concept=1 \
  --prompt_type='pet' \
  --eval_batch_size=14 \
  --num_images_per_prompt=15