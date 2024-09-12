export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUBLAS_WORKSPACE_CONFIG=:4096:8;
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/dog6";
export CUDA_VISIBLE_DEVICES=7;
accelerate launch --main_process_port 4235  db_train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<dog6>" \
  --initializer_token="sks" \
  --train_prior_concept1="dog" \
  --eval_prior_concept1="dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=1001 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="saved_models/tmp" \
  --seed=7777 \
  --mask_tokens="[MASK]" \
  --lambda_mlm=0 --freeze_mask_embedding=1 \
  --cls_net_path='saved_models/mlm_models/sd1_contextnetv4_nonpadding_1e4_unnorm_mprob015_batch150/checkpoints/checkpoint-100000/cls_net_100000_ckpt.pt' \
  --mask_embed_path='saved_models/mlm_models/sd1_contextnetv4_nonpadding_1e4_unnorm_mprob015_batch150/checkpoints/checkpoint-100000/mask_embeds_100000_ckpt.pt' \
  --mlm_target='masked' \
  --mlm_batch_size=25 \
  --run_name='tmp_nomlm_dog6' \
  --train_prompt_type='pet' \
  --eval_prompt_type='living' \
  --include_prior_concept=1 \
  --validation_steps=100 \
  --with_prior_preservation=1 \
  --class_prompt1="a picture of a dog" \
  --class_data_dir1="priors/samples_dog" \
  --simple_caption=0 \
  --caption_root='../datasets_pkgs/captions/v7' 


export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUBLAS_WORKSPACE_CONFIG=:4096:8;
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/dog6";
export CUDA_VISIBLE_DEVICES=0;
accelerate launch --main_process_port 4235  db_train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<dog6>" \
  --initializer_token="sks" \
  --train_prior_concept1="dog" \
  --eval_prior_concept1="dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=1001 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="saved_models/tmp" \
  --seed=7777 \
  --mask_tokens="[MASK]" \
  --lambda_mlm=0.001 --freeze_mask_embedding=1 \
  --cls_net_path='saved_models/mlm_models/sd1_contextnetv4_nonpadding_1e4_unnorm_mprob015_batch150/checkpoints/checkpoint-100000/cls_net_100000_ckpt.pt' \
  --mask_embed_path='saved_models/mlm_models/sd1_contextnetv4_nonpadding_1e4_unnorm_mprob015_batch150/checkpoints/checkpoint-100000/mask_embeds_100000_ckpt.pt' \
  --mlm_target='masked' \
  --mlm_batch_size=25 \
  --train_prompt_type='pet' \
  --eval_prompt_type='living' \
  --include_prior_concept=1 \
  --validation_steps=100 \
  --with_prior_preservation=1 \
  --class_prompt1="a picture of a dog" \
  --class_data_dir1="priors/samples_dog" \
  --simple_caption=0 \
  --caption_root='../datasets_pkgs/captions/v7' \
  --run_name='tmp_mlm0001_dog6' \
  --train_text_encoder 
