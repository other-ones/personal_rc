export CUBLAS_WORKSPACE_CONFIG=:4096:8;
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/cat1/";
export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUDA_VISIBLE_DEVICES=2;
accelerate launch --main_process_port 2731  generate_single.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<cat1>" \
  --resolution=512 \
  --output_dir="results/tmp/cat1" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/ti_models/old/singlev2_noprior_seed2940/cat1/tiv2_noprior_nomlm_cat1/checkpoints/learned_embeds_s3000.pt' \
  --prior_concept1='cat' \
  --include_prior_concept=0 \
  --prompt_type='pet' \
  --eval_batch_size=20 \
  --num_images_per_prompt=15  \
  --calibrate_kpos1=0 --calibrate_ppos1=0 \
  --eval_prompt_type='cat' \
  --dst_exp_path=tmp/tiv2_noprior_nomlm_cat1

export CUBLAS_WORKSPACE_CONFIG=:4096:8;
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/cat1/";
export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUDA_VISIBLE_DEVICES=3;
accelerate launch --main_process_port 2722  generate_single.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<cat1>" \
  --resolution=512 \
  --output_dir="results/tmp/cat1" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/ti_models/old/singlev2_noprior_seed2940/cat1/tiv2_noprior_mlm0001_cat1_mprob025/checkpoints/learned_embeds_s3000.pt' \
  --prior_concept1='cat' \
  --include_prior_concept=0 \
  --prompt_type='pet' \
  --eval_batch_size=20 \
  --num_images_per_prompt=15  \
  --calibrate_kpos1=0 --calibrate_ppos1=0 \
  --eval_prompt_type='cat' \
  --dst_exp_path=tmp/tiv2_noprior_mlm0001_cat1_mprob025

