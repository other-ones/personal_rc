
export DATA_DIR="/data/twkim/diffusion/personalization/dreambooth/dataset/vase"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=4;
accelerate launch --main_process_port 4812  generate_single_re_old_clsaug.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<vase>" \
  --resolution=512 \
  --output_dir="results/single/vase" \
  --seed=7677 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/single/vase/ti_noprior_no_mlm_vase_debug1/checkpoints/learned_embeds_s3000.pt' \
  --prior_concept1='vase' \
  --include_prior_concept=0 \
  --prompt_type='nonliving' \
  --eval_batch_size=20 \
  --num_images_per_prompt=20

export DATA_DIR="/data/twkim/diffusion/personalization/dreambooth/dataset/vase"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=5;
accelerate launch --main_process_port 4813  generate_single_re_old_clsaug.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<vase>" \
  --resolution=512 \
  --output_dir="results/single/vase" \
  --seed=7677 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/single/vase/ti_noprior_mlm00025_vase_debug1/checkpoints/learned_embeds_s3000.pt' \
  --prior_concept1='vase' \
  --include_prior_concept=0 \
  --prompt_type='nonliving' \
  --eval_batch_size=20 \
  --num_images_per_prompt=20

export DATA_DIR="/data/twkim/diffusion/personalization/dreambooth/dataset/vase"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=6;
accelerate launch --main_process_port 4813  generate_single_re_old_clsaug.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<vase>" \
  --resolution=512 \
  --output_dir="results/single/vase" \
  --seed=7677 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/single/vase/ti_noprior_mlm00005_vase_debug1/checkpoints/learned_embeds_s3000.pt' \
  --prior_concept1='vase' \
  --include_prior_concept=0 \
  --prompt_type='nonliving' \
  --eval_batch_size=20 \
  --num_images_per_prompt=20


  export DATA_DIR="/data/twkim/diffusion/personalization/dreambooth/dataset/vase"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=7;
accelerate launch --main_process_port 4814  generate_single_re_old_clsaug.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<vase>" \
  --resolution=512 \
  --output_dir="results/single/vase" \
  --seed=7677 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/single/vase/ti_noprior_mlm0001_vase_debug1/checkpoints/learned_embeds_s3000.pt' \
  --prior_concept1='vase' \
  --include_prior_concept=0 \
  --prompt_type='nonliving' \
  --eval_batch_size=20 \
  --num_images_per_prompt=20

  