export DATA_DIR="/data/twkim/diffusion/personalization/benchmark_dataset/decoritems_woodenpot/"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=0;
accelerate launch --main_process_port 1732  generate_single.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<dog6>" \
  --resolution=512 \
  --output_dir="results/tmp/single/wooden_pot" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/pplus_models/single/dog6/pplus_norm0_prior_mlm0001_dog6/checkpoints/learned_embeds_s3000.pt' \
  --prior_concept1='dog' \
  --include_prior_concept=1 \
  --eval_batch_size=2 \
  --prompt_type='pet' \
  --num_vectors1=9
