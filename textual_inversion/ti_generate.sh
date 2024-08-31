export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/dog6/";
export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUDA_VISIBLE_DEVICES=0;
accelerate launch --main_process_port 2731  ti_generate.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<dog6>" \
  --resolution=512 \
  --output_dir="results/tmp/dog6" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/ti_models/single_capv6_prior_seed2940_rep1/dog6/ti_cnetv4_prior_mlm0001_dog6_mprob015_mbatch25/checkpoints/learned_embeds_s3000.pt' \
  --prior_concept1='dog' \
  --include_prior_concept=1 \
  --eval_prompt_type='living' \
  --eval_batch_size=1 \
  --num_images_per_prompt=15  \
  --benchmark_path='../datasets_pkgs/eval_prompts/dreambooth.json' \
  --dst_exp_path=tmp
