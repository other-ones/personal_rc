
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/pet_dog1"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=4;
accelerate launch --main_process_port 3812  generate_single_clsaug.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<pet_dog1>" \
  --resolution=512 \
  --output_dir="results/single/pet_dog1" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/single/pet_dog1/ti_clsaug_mlm00005_pet_dog1/checkpoints/learned_embeds_s3000.pt' \
  --augmenter_path1='saved_models/single/pet_dog1/ti_clsaug_mlm00005_pet_dog1/checkpoints/augmenter_s3000.pt' \
  --prior_concept1='dog' \
  --include_prior_concept=1 \
  --prompt_type='pet' \
  --eval_batch_size=20

export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/pet_dog1"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=5;
accelerate launch --main_process_port 3813  generate_single_clsaug.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<pet_dog1>" \
  --resolution=512 \
  --output_dir="results/single/pet_dog1" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/single/pet_dog1/ti_clsaug_mlm0001_pet_dog1/checkpoints/learned_embeds_s3000.pt' \
  --augmenter_path1='saved_models/single/pet_dog1/ti_clsaug_mlm0001_pet_dog1/checkpoints/augmenter_s3000.pt' \
  --prior_concept1='dog' \
  --include_prior_concept=1 \
  --prompt_type='pet' \
  --eval_batch_size=20

export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/pet_dog1"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=6;
accelerate launch --main_process_port 3813  generate_single_clsaug.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<pet_dog1>" \
  --resolution=512 \
  --output_dir="results/single/pet_dog1" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/single/pet_dog1/ti_clsaug_mlm00025_pet_dog1/checkpoints/learned_embeds_s3000.pt' \
  --augmenter_path1='saved_models/single/pet_dog1/ti_clsaug_mlm00025_pet_dog1/checkpoints/augmenter_s3000.pt' \
  --prior_concept1='dog' \
  --include_prior_concept=0 \
  --prompt_type='pet' \
  --eval_batch_size=20

export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/pet_dog1"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=3;
accelerate launch --main_process_port 3814  generate_single_clsaug.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<pet_dog1>" \
  --resolution=512 \
  --output_dir="results/single/pet_dog1" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/single/pet_dog1/ti_clsaug_no_mlm_pet_dog1/checkpoints/learned_embeds_s3000.pt' \
  --augmenter_path1='saved_models/single/pet_dog1/ti_clsaug_no_mlm_pet_dog1/checkpoints/augmenter_s3000.pt' \
  --prior_concept1='dog' \
  --include_prior_concept=0 \
  --prompt_type='pet' \
  --eval_batch_size=20
