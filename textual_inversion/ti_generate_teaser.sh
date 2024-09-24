export CUBLAS_WORKSPACE_CONFIG=:4096:8;
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/dog6";
export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUDA_VISIBLE_DEVICES=0;
accelerate launch --main_process_port 2737  ti_generate_teaser.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<dog6>" \
  --resolution=512 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/ti_models/bigger_reduced4_prior_seed7777_qlab03_rep2/dog6/ti_bigger_qlab03_prior_mlm00001_dog6_mprob015_mbatch25_mtarget_masked/checkpoints/learned_embeds_s3000.pt' \
  --train_prior_concept1='dog' \
  --eval_prior_concept1='dog' \
  --include_prior_concept=1 \
  --eval_batch_size=15 \
  --num_images_per_prompt=30  \
  --dst_exp_path=results/ti_results/teaser/dog6

export CUBLAS_WORKSPACE_CONFIG=:4096:8;
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/dog6";
export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUDA_VISIBLE_DEVICES=1;
accelerate launch --main_process_port 2736  ti_generate_teaser.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<dog6>" \
  --resolution=512 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/ti_models/bigger_reduced4_prior_seed7777_qlab03_rep2/dog6/ti_bigger_qlab03_prior_mlm0001_dog6_mprob015_mbatch25_mtarget_masked/checkpoints/learned_embeds_s3000.pt' \
  --train_prior_concept1='dog' \
  --eval_prior_concept1='dog' \
  --include_prior_concept=1 \
  --eval_batch_size=15 \
  --num_images_per_prompt=30  \
  --dst_exp_path=results/ti_results/teaser/dog6_mlm0001

  export CUBLAS_WORKSPACE_CONFIG=:4096:8;
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/dog6";
export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUDA_VISIBLE_DEVICES=1;
accelerate launch --main_process_port 2736  ti_generate_teaser.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<dog6>" \
  --resolution=512 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/ti_models/bigger_reduced4_prior_seed7777_qlab03_rep2/dog6/ti_bigger_qlab03_prior_nomlm_dog6/checkpoints/learned_embeds_s3000.pt' \
  --train_prior_concept1='dog' \
  --eval_prior_concept1='dog' \
  --include_prior_concept=1 \
  --eval_batch_size=15 \
  --num_images_per_prompt=30  \
  --dst_exp_path=results/ti_results/teaser/dog6_nomlm