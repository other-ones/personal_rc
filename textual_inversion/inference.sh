
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="/data/twkim/diffusion/personalization/benchmark_dataset/pet_cat1"
export CUDA_VISIBLE_DEVICES=7;
accelerate launch --main_process_port 3812  inference.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<pet_cat1>" \
  --resolution=512 \
  --output_dir="inference/tiv2_no_mlm_pet_cat1_with_keyword" \
  --mask_tokens="[MASK]" \
  --prior_concept='cat' \
  --learned_embed_path='saved_models/pet_cat1/tiv2_no_mlm_pet_cat1_with_keyword/checkpoints/learned_embeds_s3000.pt' 
  

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="/data/twkim/diffusion/personalization/benchmark_dataset/pet_cat1"
export CUDA_VISIBLE_DEVICES=6;
accelerate launch --main_process_port 3813  inference.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<pet_cat1>" \
  --resolution=512 \
  --output_dir="inference/tiv2_mlm00005_pet_cat1_with_keyword_balanced" \
  --mask_tokens="[MASK]" \
  --prior_concept='cat' \
  --learned_embed_path='saved_models/pet_cat1/tiv2_mlm00005_pet_cat1_with_keyword_balanced/checkpoints/learned_embeds_s3000.pt' 
  
