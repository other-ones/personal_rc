export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/dog6"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=1;
accelerate launch --main_process_port 2731  generate_single.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<dog6>" \
  --resolution=512 \
  --output_dir="results/single/dog6" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --resume_path='saved_models/custom_diffusion/single/dog6/custom_nomlm_dog6/checkpoints/checkpoint-250' \
  --prior_concept1='dog' \
  --include_prior_concept=1 \
  --prompt_type='pet' \
  --eval_batch_size=1 \
  --num_images_per_prompt=15
