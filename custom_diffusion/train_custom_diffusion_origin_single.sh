export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export OUTPUT_DIR="outputs/pet_cat1";
export INSTANCE_DIR="/data/twkim/diffusion/personalization/collected/images/pet_cat1";
export CUDA_VISIBLE_DEVICES=0,1;
accelerate launch --main_process_port=9982 train_custom_diffusion_origin.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=./priors/samples_cat/ \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --class_prompt="cat" --num_class_images=200 \
  --instance_prompt="photo of a <new1> cat"  \
  --resolution=512  \
  --train_batch_size=2  \
  --learning_rate=1e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=251 \
  --scale_lr --hflip  \
  --modifier_token "<new1>" \
  --validation_steps=50 \
  --validation_prompt="a picture of <new1> cat" \
  --checkpointing_steps=250 \
  --no_safe_serialization

export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export OUTPUT_DIR="outputs/tmp";
export INSTANCE_DIR="/data/twkim/diffusion/personalization/collected/images/pet_cat1";
export CUDA_VISIBLE_DEVICES=2,3;
accelerate launch --main_process_port=9983 train_custom_diffusion_origin.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=./priors/samples_cat/ \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --class_prompt="cat" --num_class_images=200 \
  --instance_prompt="photo of a <new1> cat"  \
  --resolution=512  \
  --train_batch_size=2  \
  --learning_rate=1e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=251 \
  --scale_lr --hflip  \
  --modifier_token "<new1>" \
  --validation_steps=50 \
  --validation_prompt="a picture of <new1> cat" \
  --checkpointing_steps=250 \
  --no_safe_serialization
