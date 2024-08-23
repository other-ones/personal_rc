# mlm
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR1="/data/twkim/diffusion/personalization/collected/images/cat1"
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
  --output_dir="saved_models/analysis2/ti_plot_wprior" \
  --seed=7776 \
  --mask_tokens="[MASK]" \
  --freeze_mask_embedding=1 \
  --mlm_batch_size=400 \
  --prompt_type='pet' \
  --placeholder_token1="<cat1>" \
  --prior_concept1="cat" \
  --learned_embed_path1="saved_models/ti_models/singlev2_seed2940/cat1/tiv2_prior_mlm0001_cat1_mprob025/checkpoints/learned_embeds_s3000.pt" \
  --include_prior_concept=1 \
  --run_name='cat1_mlm0001' \
  --target='prior' \
  --caption_path='../datasets_pkgs/captions/analysis/pet/cat.txt'

# nomlm
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR1="/data/twkim/diffusion/personalization/collected/images/cat1"
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
  --output_dir="saved_models/analysis2/ti_plot_wprior" \
  --seed=7776 \
  --mask_tokens="[MASK]" \
  --freeze_mask_embedding=1 \
  --mlm_batch_size=400 \
  --prompt_type='pet' \
  --placeholder_token1="<cat1>" \
  --prior_concept1="cat" \
  --learned_embed_path1="saved_models/ti_models/singlev2_seed2940/cat1/tiv2_prior_nomlm_cat1/checkpoints/learned_embeds_s3000.pt" \
  --include_prior_concept=1 \
  --target='prior' \
  --run_name='cat1_nomlm' \
  --caption_path='../datasets_pkgs/captions/analysis/pet/cat.txt' 


# prior
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR1="/data/twkim/diffusion/personalization/collected/images/cat1"
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
  --output_dir="saved_models/analysis2/ti_plot_wprior" \
  --seed=7776 \
  --mask_tokens="[MASK]" \
  --freeze_mask_embedding=1 \
  --mlm_batch_size=400 \
  --prompt_type='pet' \
  --prior_concept1="cat" \
  --include_prior_concept=1 \
  --run_name='cat' \
  --target='prior' \
  --caption_path='../datasets_pkgs/captions/analysis/pet/cat_prior.txt' 

