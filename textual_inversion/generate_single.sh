export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/dog6/";
export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUDA_VISIBLE_DEVICES=0;
accelerate launch --main_process_port 2731  generate_single.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<dog6>" \
  --resolution=512 \
  --output_dir="results/tmp/dog6" \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/ti_models/singlev5s70kunnorm_noprior_seed2940_rep1/dog6/tiv5s70k_prior_mlm0001_dog6_mprob015_mbatch25_unfreeze_mask/checkpoints/learned_embeds_s500.pt' \
  --prior_concept1='dog' \
  --include_prior_concept=1 \
  --eval_prompt_type='dog' \
  --eval_batch_size=1 \
  --num_images_per_prompt=15  \
  --calibrate_kpos1=1 --calibrate_ppos1=1 \
  --rev=1 \
  --dst_exp_path=tmp
