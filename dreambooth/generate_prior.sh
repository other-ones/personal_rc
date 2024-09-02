
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./dog"
export OUTPUT_DIR="saved_models/duck"

accelerate launch generate_prior.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks duck" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 \
  --validation_steps=1 \
  --prior_concept1="duck" \
  --placeholder_token1="sks" \
  --include_prior_concept=1 \
  --train_text_encoder