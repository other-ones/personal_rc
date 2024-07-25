export DATA_DIR="/data/twkim/diffusion/personalization/benchmark_dataset/decoritems_woodenpot/"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=0;
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
  --include_prior_concept=0 \
  --eval_batch_size=20 \
  --prompt_type='nonliving' 

