ti_noprior_no_mlm_pet_cat1_re
ti_noprior_mlm00005_pet_cat1_re
ti_noprior_mlm0001_pet_cat1_re
ti_noprior_mlm00025_pet_cat1_re
export DATA_DIR="/data/twkim/diffusion/personalization/benchmark_dataset/pet_cat1/"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=0;
accelerate launch --main_process_port 1731  generate_single_re_old_clsaug.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<pet_cat1>" \
  --resolution=512 \
  --output_dir="results/single/pet_cat1" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/single/pet_cat1/ti_noprior_no_mlm_pet_cat1_re/checkpoints/learned_embeds_s3000.pt' \
  --prior_concept1='cat' \
  --include_prior_concept=0 \
  --prompt_type='pet' \
  --eval_batch_size=20 \
  --num_images_per_prompt=15 

export DATA_DIR="/data/twkim/diffusion/personalization/benchmark_dataset/pet_cat1/"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=1;
accelerate launch --main_process_port 1732  generate_single_re_old_clsaug.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<pet_cat1>" \
  --resolution=512 \
  --output_dir="results/single/pet_cat1" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/single/pet_cat1/ti_noprior_mlm00005_pet_cat1_re/checkpoints/learned_embeds_s3000.pt' \
  --prior_concept1='cat' \
  --include_prior_concept=0 \
  --prompt_type='pet' \
  --eval_batch_size=20 \
  --num_images_per_prompt=15 


export DATA_DIR="/data/twkim/diffusion/personalization/benchmark_dataset/pet_cat1/"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=2;
accelerate launch --main_process_port 1733  generate_single_re_old_clsaug.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<pet_cat1>" \
  --resolution=512 \
  --output_dir="results/single/pet_cat1" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/single/pet_cat1/ti_noprior_mlm0001_pet_cat1_re/checkpoints/learned_embeds_s3000.pt' \
  --prior_concept1='cat' \
  --include_prior_concept=0 \
  --prompt_type='pet' \
  --eval_batch_size=20 \
  --num_images_per_prompt=15 
  
export DATA_DIR="/data/twkim/diffusion/personalization/benchmark_dataset/pet_cat1/"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=3;
accelerate launch --main_process_port 1734  generate_single_re_old_clsaug.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<pet_cat1>" \
  --resolution=512 \
  --output_dir="results/single/pet_cat1" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/single/pet_cat1/ti_noprior_mlm00025_pet_cat1_re/checkpoints/learned_embeds_s3000.pt' \
  --prior_concept1='cat' \
  --include_prior_concept=0 \
  --prompt_type='pet' \
  --eval_batch_size=20 \
  --num_images_per_prompt=15 

    