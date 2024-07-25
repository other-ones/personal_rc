
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/pet_cat1"
export CUDA_VISIBLE_DEVICES=0;
accelerate launch textual_inversion_train_mlm.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=10001 \
  --learning_rate=5e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --seed=7777 \
  --mask_tokens="[MASK]" \
  --lambda_mlm=0 \
  --output_dir='saved_models/pet_cat1' \
  --run_name="tiv2_no_mlm_pet_cat1_with_keyword_s10000" \
  --scale_lr \
  --placeholder_token="<pet_cat1>" \
  --initializer_token="cat" \
  --prompt_type='pet' \
  --prior_concept="cat" \
  --include_prior_concept=1 \
  --report_to='wandb' \
  --project_name='TI MLM V2' 



