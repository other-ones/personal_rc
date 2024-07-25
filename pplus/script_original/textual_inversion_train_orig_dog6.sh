export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/dog6"
export CUDA_VISIBLE_DEVICES=2;
accelerate launch textual_inversion_train_mlm.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<dog6>" \
  --initializer_token="dog" \
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
  --output_dir='saved_models/dog6' \
  --run_name="tiv2_no_mlm_dog6_with_keyword" \
  --scale_lr \
  --prompt_type='pet' \
  --prior_concept="dog" \
  --include_prior_concept=1 \
  --report_to='wandb' \
  --project_name='TI MLM V2' 

