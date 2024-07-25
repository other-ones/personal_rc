export DATA_DIR="/data/twkim/diffusion/personalization/benchmark_dataset/decoritems_woodenpot/"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=4;
accelerate launch --main_process_port 1732  generate_single.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<wooden_pot>" \
  --resolution=512 \
  --output_dir="results/single/wooden_pot" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/single/wooden_pot/ti_mlm0001_wooden_pot_unfreezemask/checkpoints/learned_embeds_s3000.pt' \
  --prior_concept1='wooden pot' \
  --include_prior_concept=1 \
  --eval_batch_size=20 \
  --prompt_type='nonliving' 



export DATA_DIR="/data/twkim/diffusion/personalization/benchmark_dataset/decoritems_woodenpot/"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=5;
accelerate launch --main_process_port 1731  generate_single.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<wooden_pot>" \
  --resolution=512 \
  --output_dir="results/single/wooden_pot" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/single/wooden_pot/ti_mlm00005_wooden_pot_unfreezemask/checkpoints/learned_embeds_s3000.pt' \
  --prior_concept1='wooden pot' \
  --include_prior_concept=1 \
  --eval_batch_size=20 \
  --prompt_type='nonliving' 

  export DATA_DIR="/data/twkim/diffusion/personalization/benchmark_dataset/decoritems_woodenpot/"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=6;
accelerate launch --main_process_port 1731  generate_single.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<wooden_pot>" \
  --resolution=512 \
  --output_dir="results/single/wooden_pot" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/single/wooden_pot/ti_mlm00005_wooden_pot_unfreezemask_masked/checkpoints/learned_embeds_s3000.pt' \
  --prior_concept1='wooden pot' \
  --include_prior_concept=1 \
  --eval_batch_size=20 \
  --prompt_type='nonliving' 

    export DATA_DIR="/data/twkim/diffusion/personalization/benchmark_dataset/decoritems_woodenpot/"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=7;
accelerate launch --main_process_port 1731  generate_single.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<wooden_pot>" \
  --resolution=512 \
  --output_dir="results/single/wooden_pot" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/single/wooden_pot/ti_mlm00025_wooden_pot_unfreezemask/checkpoints/learned_embeds_s3000.pt' \
  --prior_concept1='wooden pot' \
  --include_prior_concept=1 \
  --eval_batch_size=20 \
  --prompt_type='nonliving' 