
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/dog6/";
export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUDA_VISIBLE_DEVICES=7;
accelerate launch --main_process_port 2731  visualize_attns_ce.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<dog6>" \
  --resolution=512 \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/ti_attn_models/bigger_reduced4_prior_seed7777_qlab03_rep2/dog6/ti_bigger_qlab03_prior_nomlm_dog6/checkpoints/learned_embeds_s2000.pt' \
  --train_prior_concept1='dog' \
  --eval_prior_concept1='dog' \
  --include_prior_concept=1 \
  --prompt_type='pet' \
  --eval_batch_size=5 \
  --num_images_per_prompt=1  \
  --dst_exp_path=cross_attns/dog6/ti_bigger_qlab03_prior_nomlm_dog6_s2000 \
  --benchmark_path='../datasets_pkgs/eval_prompts/analysis_ce.json' \
  --eval_prompt_type="all" 

export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/dog6/";
export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUDA_VISIBLE_DEVICES=6;
accelerate launch --main_process_port 2732  visualize_attns_ce.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<dog6>" \
  --resolution=512 \
  --seed=1234 \
  --mask_tokens="[MASK]" \
  --learned_embed_path1='saved_models/ti_attn_models/bigger_reduced4_prior_seed7777_qlab03_rep2/dog6/ti_bigger_qlab03_prior_mlm0001_dog6_mprob015_mbatch25_mtarget_masked/checkpoints/learned_embeds_s2000.pt' \
  --train_prior_concept1='dog' \
  --eval_prior_concept1='dog' \
  --include_prior_concept=1 \
  --prompt_type='pet' \
  --eval_batch_size=1 \
  --num_images_per_prompt=1  \
  --dst_exp_path=cross_attns/dog6/ti_bigger_qlab03_prior_mlm0001_dog6_mprob015_mbatch25_mtarget_masked_s2000 \
  --benchmark_path='../datasets_pkgs/eval_prompts/analysis_ce.json' \
  --eval_prompt_type="all" 

