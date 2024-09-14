export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/pet_dog1/";
export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUDA_VISIBLE_DEVICES=1;
accelerate launch --main_process_port 1732  pplus_generate_clean.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<pet_dog1>" \
  --resolution=512 \
  --output_dir="tmp" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/pplus_models/tmp_pplus_mlm00001_lr5e4/checkpoints/learned_embeds_s2926.pt' \
  --prior_concept1='dog' \
  --include_prior_concept=0 \
  --eval_batch_size=1 \
  --eval_prompt_type='living' \
  --num_vectors1=7 \
  --benchmark_path='../datasets_pkgs/eval_prompts/dreambooth.json' \
