export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/dog6"
export CUDA_VISIBLE_DEVICES=0;
accelerate launch --main_process_port 4235  train_dreambooth_single_mlm.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --placeholder_token1="<dog6>" \
  --prior_concept1="dog" \
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
  --cls_net_path='saved_models/mlm_models/sd1_contextnetv2_nonpadding_1e4/checkpoints/checkpoint-60000/cls_net_60000_ckpt.pt' \
  --mask_embed_path='saved_models/mlm_models/sd1_contextnetv2_nonpadding_1e4/checkpoints/checkpoint-60000/mask_embeds_60000_ckpt.pt' \
  --mlm_target='masked' \
  --mlm_batch_size=20 \
  --run_name='tmp_mlm0005_dog6' \
  --prompt_type='pet' \
  --include_prior_concept=1 \
  --train_text_encoder \
  --validation_steps=100 \
  --with_prior_preservation=1 \
  --class_prompt1="a picture of a dog" \
  --class_data_dir1="priors/samples_dog" \
  --simple_caption=0 \
  --make_composition=0 \
  --caption_root='../datasets_pkgs/captions'