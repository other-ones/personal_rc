export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/teapot"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=1;
export CUBLAS_WORKSPACE_CONFIG=:4096:8;
accelerate launch --main_process_port 2731  cd_generate.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<teapot>" \
  --resolution=512 \
  --output_dir="tmp" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --resume_cd_path='saved_models/cd_models/single_capv7_seed2940_rep1/teapot/cd_cnetv4_mlm0001_teapot_mprob015_mbatch25_lr1e5/checkpoints/checkpoint-0/custom_diffusion.pt' \
  --learned_embed_path1='saved_models/cd_models/single_capv7_seed2940_rep1/teapot/cd_cnetv4_mlm0001_teapot_mprob015_mbatch25_lr1e5/checkpoints/checkpoint-0/learned_embeds.pt' \
  --train_prior_concept1='teapot' \
  --eval_prior_concept1='teapot' \
  --include_prior_concept=1 \
  --eval_prompt_type='nonliving' \
  --eval_batch_size=1 \
  --num_images_per_prompt=15 \
  --dst_exp_path=tmp \
  --benchmark_path='../datasets_pkgs/eval_prompts/dreambooth.json'
