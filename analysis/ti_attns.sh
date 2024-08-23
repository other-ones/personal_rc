export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR1="/data/twkim/diffusion/personalization/collected/images/dog6"
export CUDA_VISIBLE_DEVICES=3;
accelerate launch --main_process_port 1234  ti_embeddings.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR1 \
  --resolution=512 \
  --train_batch_size=1 \
  --max_train_steps=3001 \
  --learning_rate=5e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="saved_models/analysis/ti_plot_noprior" \
  --seed=7776 \
  --mask_tokens="[MASK]" \
  --freeze_mask_embedding=1 \
  --mlm_target='masked' \
  --mlm_batch_size=400 \
  --prompt_type='pet' \
  --placeholder_token1="<dog6>" \
  --prior_concept1="dog" \
  --learned_embed_path1="saved_models/ti_models/singlev2_seed2940/dog6/tiv2_prior_nomlm_dog6/checkpoints/learned_embeds_s3000.pt" \
  --include_prior_concept=1 \
  --run_name='dog6_nomlm' \
  --include_keyword=1 \
  --caption_root='../datasets_pkgs/captions/analysis'



export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR1="/data/twkim/diffusion/personalization/collected/images/dog6"
export CUDA_VISIBLE_DEVICES=2;
accelerate launch --main_process_port 1234  ti_embeddings.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR1 \
  --resolution=512 \
  --train_batch_size=1 \
  --max_train_steps=3001 \
  --learning_rate=5e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="saved_models/analysis/ti_plot_noprior" \
  --seed=7776 \
  --mask_tokens="[MASK]" \
  --freeze_mask_embedding=1 \
  --mlm_target='masked' \
  --mlm_batch_size=400 \
  --prompt_type='pet' \
  --placeholder_token1="<dog6>" \
  --prior_concept1="dog" \
  --include_prior_concept=1 \
  --run_name='dog' \
  --include_keyword=0 \
  --caption_root='../datasets_pkgs/captions/analysis'


export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR1="/data/twkim/diffusion/personalization/collected/images/dog6"
export CUDA_VISIBLE_DEVICES=1;
accelerate launch --main_process_port 1234  ti_embeddings.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR1 \
  --resolution=512 \
  --train_batch_size=1 \
  --max_train_steps=3001 \
  --learning_rate=5e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="saved_models/analysis/ti_plot_noprior" \
  --seed=7776 \
  --mask_tokens="[MASK]" \
  --freeze_mask_embedding=1 \
  --mlm_target='masked' \
  --mlm_batch_size=400 \
  --prompt_type='pet' \
  --placeholder_token1="<dog6>" \
  --prior_concept1="dog" \
  --learned_embed_path1="saved_models/ti_models/singlev2_seed2940/dog6/tiv2_prior_mlm0001_dog6_mprob025/checkpoints/learned_embeds_s3000.pt" \
  --include_prior_concept=1 \
  --run_name='dog6_mlm0001' \
  --include_keyword=1 \
  --caption_root='../datasets_pkgs/captions/analysis'