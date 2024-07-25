export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/teddybear"
export CUDA_VISIBLE_DEVICES=0;
accelerate launch --main_process_port 8374  textual_inversion_train_mlm_triplet.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3001 \
  --learning_rate=5e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="saved_models/teddybear" \
  --seed=7777 \
  --mask_tokens="[MASK]" \
  --lambda_mlm=0.0005 --freeze_mask_embedding=1 \
  --cls_net_path='saved_models/mlm_contextnet_nonpad_lr1e4/checkpoints/cls_net_99000_ckpt.pt' \
  --mask_embed_path='saved_models/mlm_contextnet_nonpad_lr1e4/checkpoints/mask_embeds_99000_ckpt.pt' \
  --mlm_target='masked' \
  --mlm_batch_size=50 \
  --run_name='tmp' \
  --scale_lr \
  --placeholder_token="<teddybear>" \
  --initializer_token="teddybear" \
  --prompt_type='nonliving' \
  --prior_concept="teddybear" \
  --include_prior_concept=1 \
  --project_name='TI MLM V3 Triplet' 

  
