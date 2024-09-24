export CUBLAS_WORKSPACE_CONFIG=:4096:8;
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/teapot";
export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUDA_VISIBLE_DEVICES=1;
accelerate launch --main_process_port 2736  ti_generate_teaser.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<teapot>" \
  --resolution=512 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/ti_models/bigger_reduced4_prior_seed7777_qlab03_rep2/teapot/ti_bigger_qlab03_prior_mlm0001_teapot_mprob015_mbatch25_mtarget_masked/checkpoints/learned_embeds_s3000.pt' \
  --train_prior_concept1='teapot' \
  --eval_prior_concept1='teapot' \
  --include_prior_concept=1 \
  --eval_batch_size=1 \
  --num_images_per_prompt=30  \
  --teaser_prompt="a picture of {} at Times Square" \
  --dst_exp_path=results/ti_results/teaser/teapot_mlm0001_timesquare

  export CUBLAS_WORKSPACE_CONFIG=:4096:8;
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/teapot";
export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUDA_VISIBLE_DEVICES=1;
accelerate launch --main_process_port 2736  ti_generate_teaser.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<teapot>" \
  --resolution=512 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/ti_models/bigger_reduced4_prior_seed7777_qlab03_rep2/teapot/ti_bigger_qlab03_prior_nomlm_teapot/checkpoints/learned_embeds_s3000.pt' \
  --train_prior_concept1='teapot' \
  --eval_prior_concept1='teapot' \
  --include_prior_concept=1 \
  --eval_batch_size=1 \
  --num_images_per_prompt=30  \
  --teaser_prompt="a picture of {} at Times Square" \
  --dst_exp_path=results/ti_results/teaser/teapot_nomlm_timesquare