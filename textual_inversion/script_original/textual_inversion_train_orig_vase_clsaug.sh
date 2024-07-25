
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/vase"
export CUDA_VISIBLE_DEVICES=0;
accelerate launch textual_inversion_train_mlm_clsaug.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token1="<vase>" \
  --prior_concept1="vase" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3001 \
  --learning_rate=5e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --seed=7777 \
  --mask_tokens="[MASK]" \
  --lambda_mlm=0 \
  --output_dir='saved_models/single/vase' \
  --run_name="ti_clsaug_no_mlm_vase" \
  --scale_lr \
  --prompt_type='nonliving' \
  --include_prior_concept=0 