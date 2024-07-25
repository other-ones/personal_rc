export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/barn"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=6;
accelerate launch --main_process_port 4814  generate_single.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<barn>" \
  --resolution=512 \
  --output_dir="results/single/barn" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/single/barn/ti_mlm0001_barn_unfreezemask/checkpoints/learned_embeds_s3000.pt' \
  --prior_concept1='barn' \
  --include_prior_concept=1 \
  --prompt_type='building' \
  --eval_batch_size=20

  export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/barn"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=0;
accelerate launch --main_process_port 4814  generate_single.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<barn>" \
  --resolution=512 \
  --output_dir="results/single/barn" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/single/barn/ti_mlm00005_barn_unfreezemask/checkpoints/learned_embeds_s3000.pt' \
  --prior_concept1='barn' \
  --include_prior_concept=1 \
  --prompt_type='building' \
  --eval_batch_size=20


  export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/barn"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=2;
accelerate launch --main_process_port 4815  generate_single.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<barn>" \
  --resolution=512 \
  --output_dir="results/single/barn" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/single/barn/ti_mlm00005_barn_unfreezemask_masked/checkpoints/learned_embeds_s3000.pt' \
  --prior_concept1='barn' \
  --include_prior_concept=1 \
  --prompt_type='building' \
  --eval_batch_size=20



  export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/barn"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=1;
accelerate launch --main_process_port 4815  generate_single.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<barn>" \
  --resolution=512 \
  --output_dir="results/single/barn" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/single/barn/ti_mlm00025_barn_unfreezemask/checkpoints/learned_embeds_s3000.pt' \
  --prior_concept1='barn' \
  --include_prior_concept=1 \
  --prompt_type='building' \
  --eval_batch_size=20
