export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/rc_car"
export CUDA_VISIBLE_DEVICES=0;
accelerate launch textual_inversion_train_mlm2.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token1="<rc_car>" \
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
  --output_dir='saved_models/single/rc_car' \
  --run_name="tiv2_no_mlm_rc_car" \
  --scale_lr \
  --prompt_type='nonliving' \
  --prior_concept1="toy" \
  --include_prior_concept=1 \
  --project_name='TI MLM V3 Triplet' 

