
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/vase"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=0;
accelerate launch --main_process_port 3812  generate_single_clsaug.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<vase>" \
  --resolution=512 \
  --output_dir="results/single/vase" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/single/vase/ti_clsaug_mlm00005_vase/checkpoints/learned_embeds_s3000.pt' \
  --augmenter_path1='saved_models/single/vase/ti_clsaug_mlm00005_vase/checkpoints/augmenter_s3000.pt' \
  --prior_concept1='vase' \
  --include_prior_concept=1 \
  --prompt_type='nonliving' \
  --eval_batch_size=20

export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/vase"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=1;
accelerate launch --main_process_port 3813  generate_single_clsaug.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<vase>" \
  --resolution=512 \
  --output_dir="results/single/vase" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/single/vase/ti_clsaug_mlm0001_vase/checkpoints/learned_embeds_s3000.pt' \
  --augmenter_path1='saved_models/single/vase/ti_clsaug_mlm0001_vase/checkpoints/augmenter_s3000.pt' \
  --prior_concept1='vase' \
  --include_prior_concept=1 \
  --prompt_type='nonliving' \
  --eval_batch_size=20

export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/vase"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=2;
accelerate launch --main_process_port 3813  generate_single_clsaug.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<vase>" \
  --resolution=512 \
  --output_dir="results/single/vase" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/single/vase/ti_clsaug_mlm00025_vase/checkpoints/learned_embeds_s3000.pt' \
  --augmenter_path1='saved_models/single/vase/ti_clsaug_mlm00025_vase/checkpoints/augmenter_s3000.pt' \
  --prior_concept1='vase' \
  --include_prior_concept=0 \
  --prompt_type='nonliving' \
  --eval_batch_size=20

export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/vase"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=3;
accelerate launch --main_process_port 3814  generate_single_clsaug.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<vase>" \
  --resolution=512 \
  --output_dir="results/single/vase" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/single/vase/ti_clsaug_no_mlm_vase/checkpoints/learned_embeds_s3000.pt' \
  --augmenter_path1='saved_models/single/vase/ti_clsaug_no_mlm_vase/checkpoints/augmenter_s3000.pt' \
  --prior_concept1='vase' \
  --include_prior_concept=0 \
  --prompt_type='nonliving' \
  --eval_batch_size=20
