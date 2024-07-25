export DATA_DIR="/data/twkim/diffusion/personalization/dreambooth/dataset/backpack/"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=2;
accelerate launch --main_process_port 1732  generate_single.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<backpack>" \
  --resolution=512 \
  --output_dir="results/single/backpack" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/single/backpack/ti_mlm0001_backpack_unfreezemask/checkpoints/learned_embeds_s3000.pt' \
  --prior_concept1='backpack' \
  --include_prior_concept=1 \
  --prompt_type='nonliving' \
  --eval_batch_size=20

export DATA_DIR="/data/twkim/diffusion/personalization/dreambooth/dataset/backpack/"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=6;
accelerate launch --main_process_port 1733  generate_single.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<backpack>" \
  --resolution=512 \
  --output_dir="results/single/backpack" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/single/backpack/ti_mlm00005_backpack_unfreezemask/checkpoints/learned_embeds_s3000.pt' \
  --prior_concept1='backpack' \
  --include_prior_concept=1 \
  --prompt_type='nonliving' \
  --eval_batch_size=20


export DATA_DIR="/data/twkim/diffusion/personalization/dreambooth/dataset/backpack/"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=6;
accelerate launch --main_process_port 1733  generate_single.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<backpack>" \
  --resolution=512 \
  --output_dir="results/single/backpack" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/single/backpack/ti_mlm00005_backpack_unfreezemask_masked/checkpoints/learned_embeds_s3000.pt' \
  --prior_concept1='backpack' \
  --include_prior_concept=1 \
  --prompt_type='nonliving' \
  --eval_batch_size=20



export DATA_DIR="/data/twkim/diffusion/personalization/dreambooth/dataset/backpack/"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=0;
accelerate launch --main_process_port 1733  generate_single.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<backpack>" \
  --resolution=512 \
  --output_dir="results/single/backpack" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/single/backpack/ti_mlm00025_backpack_unfreezemask/checkpoints/learned_embeds_s3000.pt' \
  --prior_concept1='backpack' \
  --include_prior_concept=1 \
  --prompt_type='nonliving' \
  --eval_batch_size=20