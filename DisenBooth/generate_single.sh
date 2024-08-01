export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/dog6";
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base";
export CUDA_VISIBLE_DEVICES=0;
accelerate launch --main_process_port 2731  generate_single.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<dog6>" \
  --resolution=512 \
  --output_dir="results/disenbooth/sd2/single/dog6" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --resume_unet_path='saved_models/disenbooth_models/sd2/single/dog6/disenbooth_nopp_mlm0001_dog6_train_text_lr2e5_alr1e4/checkpoints/checkpoint-3000/unet_5000.pt' \
  --learned_embed_path1='saved_models/disenbooth_models/sd2/single/dog6/disenbooth_nopp_mlm0001_dog6_train_text_lr2e5_alr1e4/checkpoints/checkpoint-3000/learned_embeds_3000.pt' \
  --prior_concept1='dog' \
  --include_prior_concept=1 \
  --prompt_type='pet' \
  --eval_batch_size=5 \
  --num_images_per_prompt=15