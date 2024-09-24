  
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/teapot"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=0;
export CUBLAS_WORKSPACE_CONFIG=:4096:8;
accelerate launch --main_process_port 2731  cd_generate_teaser.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<teapot>" \
  --resolution=512 \
  --output_dir="tmp" \
  --mask_tokens="[MASK]" \
  --resume_cd_path='saved_models/cd_models/init_seed2940_qlab03_rep2/teapot/cd_init_qlab03_nomlm_teapot_lr1e5/checkpoints/checkpoint-500/custom_diffusion.pt' \
  --learned_embed_path1='saved_models/cd_models/init_seed2940_qlab03_rep2/teapot/cd_init_qlab03_nomlm_teapot_lr1e5/checkpoints/checkpoint-500/learned_embeds.pt' \
  --train_prior_concept1='teapot' \
  --eval_prior_concept1='teapot' \
  --include_prior_concept=1 \
  --eval_batch_size=1 \
  --num_images_per_prompt=30 \
  --dst_exp_path=results/cd_results/teaser/teapot_nomlm_acropolis \
  --teaser_prompt="a picture of {} at Times Square"
  
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/teapot"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=1;
export CUBLAS_WORKSPACE_CONFIG=:4096:8;
accelerate launch --main_process_port 2731  cd_generate_teaser.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<teapot>" \
  --resolution=512 \
  --output_dir="tmp" \
  --mask_tokens="[MASK]" \
  --resume_cd_path='saved_models/cd_models/highmlm_seed2940_qlab03_rep1/teapot/cd_highmlm_qlab03_mlm01_teapot_mprob015_mbatch25_mtarget_masked_lr1e5/checkpoints/checkpoint-500/custom_diffusion.pt' \
  --learned_embed_path1='saved_models/cd_models/highmlm_seed2940_qlab03_rep1/teapot/cd_highmlm_qlab03_mlm01_teapot_mprob015_mbatch25_mtarget_masked_lr1e5/checkpoints/checkpoint-500/learned_embeds.pt' \
  --train_prior_concept1='teapot' \
  --eval_prior_concept1='teapot' \
  --include_prior_concept=1 \
  --eval_batch_size=1 \
  --num_images_per_prompt=30 \
  --dst_exp_path=results/cd_results/teaser/teapot_mlm01_acropolis \
  --teaser_prompt="a picture of {} at Times Square"