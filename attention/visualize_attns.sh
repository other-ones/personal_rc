export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/cat1/";
export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUDA_VISIBLE_DEVICES=1;
accelerate launch --main_process_port 2732  visualize_attns.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<cat1>" \
  --resolution=512 \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/ti_models/singlev2_re_noprior_seed2940/cat1/tiv2_re_noprior_mlm0001_cat1_mprob025/checkpoints/learned_embeds_s3000.pt' \
  --prior_concept1='cat' \
  --include_prior_concept=1 \
  --prompt_type='pet' \
  --eval_batch_size=1 \
  --num_images_per_prompt=1  \
  --dst_exp_path=cross_attns/cat1/tiv2_re_noprior_mlm0001_cat1_mprob025


export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/cat1/";
export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUDA_VISIBLE_DEVICES=2;
accelerate launch --main_process_port 2731  visualize_attns.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<cat1>" \
  --resolution=512 \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/ti_models/singlev2_re_noprior_seed2940/cat1/tiv2_re_noprior_nomlm_cat1/checkpoints/learned_embeds_s3000.pt' \
  --prior_concept1='cat' \
  --include_prior_concept=1 \
  --prompt_type='pet' \
  --eval_batch_size=1 \
  --num_images_per_prompt=1  \
  --dst_exp_path=cross_attns/cat1/tiv2_re_noprior_mlm0001_cat1_mprob025
