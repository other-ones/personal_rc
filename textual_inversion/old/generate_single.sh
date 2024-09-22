export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/teapot/";
export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUDA_VISIBLE_DEVICES=2;
accelerate launch --main_process_port 2731  ti_generate.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<teapot>" \
  --resolution=512 \
  --output_dir="results/tmp/teapot" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/ti_models/single_capv7_prior_seed2940_rep2/teapot/ti_cnetv4_prior_mlm0001_teapot_mprob015_mbatch25/checkpoints/learned_embeds_s3000.pt' \
  --train_prior_concept1='teapot' \
  --eval_prior_concept1='teapot' \
  --include_prior_concept=1 \
  --eval_prompt_type='nonliving' \
  --eval_batch_size=1 \
  --num_images_per_prompt=15  \
  --calibrate_kpos1=1 --calibrate_ppos1=1 \
  --benchmark_path='../datasets_pkgs/eval_prompts/dreambooth.json'  \
  --dst_exp_path=results/ti_results/tmp_teapot_rep1


export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/teapot/";
export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUDA_VISIBLE_DEVICES=3;
accelerate launch --main_process_port 2733  ti_generate.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<teapot>" \
  --resolution=512 \
  --output_dir="results/tmp/teapot" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/ti_models/single_capv7_prior_seed2940_rep2/teapot/ti_cnetv4_prior_mlm0001_teapot_mprob015_mbatch25/checkpoints/learned_embeds_s3000.pt' \
  --train_prior_concept1='teapot' \
  --eval_prior_concept1='teapot' \
  --include_prior_concept=1 \
  --eval_prompt_type='nonliving' \
  --eval_batch_size=1 \
  --num_images_per_prompt=15  \
  --calibrate_kpos1=1 --calibrate_ppos1=1 \
  --benchmark_path='../datasets_pkgs/eval_prompts/dreambooth.json'  \
  --dst_exp_path=results/ti_results/tmp_teapot_rep2
