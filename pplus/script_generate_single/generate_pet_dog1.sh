export DATA_DIR="/data/twkim/diffusion/personalization/benchmark_dataset/pet_dog1/"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=4;
accelerate launch --main_process_port 2731  generate_single_re_old_clsaug.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<pet_dog1>" \
  --resolution=512 \
  --output_dir="results/single/pet_dog1" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/single/pet_dog1/ti_noprior_no_mlm_pet_dog1_re/checkpoints/learned_embeds_s3000.pt' \
  --prior_concept1='dog' \
  --include_prior_concept=0 \
  --prompt_type='pet' \
  --eval_batch_size=20 \
  --num_images_per_prompt=15 

export DATA_DIR="/data/twkim/diffusion/personalization/benchmark_dataset/pet_dog1/"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=5;
accelerate launch --main_process_port 2732  generate_single_re_old_clsaug.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<pet_dog1>" \
  --resolution=512 \
  --output_dir="results/single/pet_dog1" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/single/pet_dog1/ti_noprior_mlm00005_pet_dog1_re/checkpoints/learned_embeds_s3000.pt' \
  --prior_concept1='dog' \
  --include_prior_concept=0 \
  --prompt_type='pet' \
  --eval_batch_size=20 \
  --num_images_per_prompt=15 


export DATA_DIR="/data/twkim/diffusion/personalization/benchmark_dataset/pet_dog1/"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=6;
accelerate launch --main_process_port 2733  generate_single_re_old_clsaug.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<pet_dog1>" \
  --resolution=512 \
  --output_dir="results/single/pet_dog1" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/single/pet_dog1/ti_noprior_mlm0001_pet_dog1_re/checkpoints/learned_embeds_s3000.pt' \
  --prior_concept1='dog' \
  --include_prior_concept=0 \
  --prompt_type='pet' \
  --eval_batch_size=20 \
  --num_images_per_prompt=15 
  
export DATA_DIR="/data/twkim/diffusion/personalization/benchmark_dataset/pet_dog1/"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=7;
accelerate launch --main_process_port 2734  generate_single_re_old_clsaug.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<pet_dog1>" \
  --resolution=512 \
  --output_dir="results/single/pet_dog1" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/single/pet_dog1/ti_noprior_mlm00025_pet_dog1_re/checkpoints/learned_embeds_s3000.pt' \
  --prior_concept1='dog' \
  --include_prior_concept=0 \
  --prompt_type='pet' \
  --eval_batch_size=20 \
  --num_images_per_prompt=15 

    