export DATA_DIR1="/data/twkim/diffusion/personalization/collected/images/pet_cat1/"
export DATA_DIR2="/data/twkim/diffusion/personalization/collected/images/pet_dog1/"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=2;
accelerate launch --main_process_port 2732  generate_multi.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR1 \
  --train_data_dir2=$DATA_DIR2 \
  --resolution=512 \
  --output_dir="results/multi/two_pets" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --prior_concept1='cat' \
  --prior_concept2='dog' \
  --include_prior_concept=0 \
  --prompt_type='two_pets' \
  --eval_batch_size=20 \
  --placeholder_token1="<pet_cat1>" \
  --placeholder_token2="<pet_dog1>" \
  --learned_embed_path_multi='saved_models/multi/two_pets/pet_cat1_pet_dog1_nocompose_mask_mlm0001_nocontrast_lr5e4/checkpoints/learned_embeds_multi_s3000.pt' \
  --num_images_per_prompt=15 


export DATA_DIR1="/data/twkim/diffusion/personalization/collected/images/pet_cat1/"
export DATA_DIR2="/data/twkim/diffusion/personalization/collected/images/pet_dog1/"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=3;
accelerate launch --main_process_port 2733  generate_multi.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR1 \
  --train_data_dir2=$DATA_DIR2 \
  --resolution=512 \
  --output_dir="results/multi/two_pets" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --prior_concept1='cat' \
  --prior_concept2='dog' \
  --include_prior_concept=0 \
  --prompt_type='two_pets' \
  --eval_batch_size=20 \
  --placeholder_token1="<pet_cat1>" \
  --placeholder_token2="<pet_dog1>" \
  --learned_embed_path_multi='saved_models/multi/two_pets/pet_cat1_pet_dog1_nocompose_nomask_mlm0001_nocontrast_lr5e4/checkpoints/learned_embeds_multi_s3000.pt' \
  --num_images_per_prompt=15 




export DATA_DIR1="/data/twkim/diffusion/personalization/collected/images/pet_cat1/"
export DATA_DIR2="/data/twkim/diffusion/personalization/collected/images/pet_dog1/"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=4;
accelerate launch --main_process_port 2733  generate_multi.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR1 \
  --train_data_dir2=$DATA_DIR2 \
  --resolution=512 \
  --output_dir="results/multi/two_pets" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --prior_concept1='cat' \
  --prior_concept2='dog' \
  --include_prior_concept=0 \
  --prompt_type='two_pets' \
  --eval_batch_size=20 \
  --placeholder_token1="<pet_cat1>" \
  --placeholder_token2="<pet_dog1>" \
  --learned_embed_path_multi='saved_models/multi/two_pets/pet_cat1_pet_dog1_nomlm_contrast0005_lr5e4/checkpoints/learned_embeds_multi_s3000.pt' \
  --num_images_per_prompt=15 




export DATA_DIR1="/data/twkim/diffusion/personalization/collected/images/pet_cat1/"
export DATA_DIR2="/data/twkim/diffusion/personalization/collected/images/pet_dog1/"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=5;
accelerate launch --main_process_port 2735  generate_multi.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR1 \
  --train_data_dir2=$DATA_DIR2 \
  --resolution=512 \
  --output_dir="results/multi/two_pets" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --prior_concept1='cat' \
  --prior_concept2='dog' \
  --include_prior_concept=0 \
  --prompt_type='two_pets' \
  --eval_batch_size=20 \
  --placeholder_token1="<pet_cat1>" \
  --placeholder_token2="<pet_dog1>" \
  --learned_embed_path_multi='saved_models/multi/two_pets/pet_cat1_pet_dog1_nomlm_contrast0001_lr5e4/checkpoints/learned_embeds_multi_s3000.pt' \
  --num_images_per_prompt=15 










export DATA_DIR1="/data/twkim/diffusion/personalization/collected/images/pet_cat1/"
export DATA_DIR2="/data/twkim/diffusion/personalization/collected/images/pet_dog1/"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=7;
accelerate launch --main_process_port 2737  generate_multi.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR1 \
  --train_data_dir2=$DATA_DIR2 \
  --resolution=512 \
  --output_dir="results/multi/two_pets" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --prior_concept1='cat' \
  --prior_concept2='dog' \
  --include_prior_concept=0 \
  --prompt_type='two_pets' \
  --eval_batch_size=20 \
  --placeholder_token1="<pet_cat1>" \
  --placeholder_token2="<pet_dog1>" \
  --learned_embed_path_multi='saved_models/multi/two_pets/pet_cat1_pet_dog1_nomlm_contrast00001_lr5e4/checkpoints/learned_embeds_multi_s3000.pt' \
  --num_images_per_prompt=15 





























# single
export DATA_DIR1="/data/twkim/diffusion/personalization/collected/images/pet_cat1/"
export DATA_DIR2="/data/twkim/diffusion/personalization/collected/images/pet_dog1/"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=2;
accelerate launch --main_process_port 2731  generate_multi.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR1 \
  --train_data_dir2=$DATA_DIR2 \
  --resolution=512 \
  --output_dir="results/multi/two_pets" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --prior_concept1='cat' \
  --prior_concept2='dog' \
  --include_prior_concept=0 \
  --prompt_type='two_pets' \
  --eval_batch_size=2 \
  --placeholder_token1="<pet_cat1>" \
  --placeholder_token2="<pet_dog1>" \
  --learned_embed_path1='saved_models/tmp/pet_cat1/ti_norm8_noprior_mlm0001_pet_cat1/checkpoints/learned_embeds_s3000.pt' \
  --learned_embed_path2='saved_models/tmp/pet_dog1/ti_norm8_noprior_mlm0001_pet_dog1/checkpoints/learned_embeds_s3000.pt' \
  --exp_name='tmp' \
  --normalize_target1=2 \
  --normalize_target2=2 \
  --num_images_per_prompt=15 