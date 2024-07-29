export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/dog6";
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base";
export CUDA_VISIBLE_DEVICES=6;
accelerate launch --main_process_port 2731  generate_single.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<dog6>" \
  --resolution=512 \
  --output_dir="results/disenbooth/sd2/single/dog6" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --resume_lora_path='saved_models/disenbooth_models/tmp_mlm0001_dog6/checkpoints/checkpoint-0600' \
  --prior_concept1='cat' \
  --include_prior_concept=1 \
  --prompt_type='pet' \
  --eval_batch_size=20 \
  --num_images_per_prompt=15