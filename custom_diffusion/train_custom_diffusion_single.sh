export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export OUTPUT_DIR="saved_models/custom_diffusion/single";
export INSTANCE_DIR="/data/twkim/diffusion/personalization/collected/images/pet_cat1";
export CUDA_VISIBLE_DEVICES=6,7;
accelerate launch train_custom_diffusion_single.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir1=./priors/samples_cat/ \
  --validation_prompt="photo of a <pet_cat1> cat"  \
  --resolution=512  \
  --train_batch_size=4  \
  --learning_rate=8e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=250 \
  --scale_lr --hflip  \
  --with_prior_preservation=1 --prior_loss_weight=1.0 \
  --class_prompt1="cat" --num_class_images=200 \
  --placeholder_token1 "<pet_cat1>"
  # --real_prior


export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export OUTPUT_DIR="saved_models/custom_diffusion/single";
export INSTANCE_DIR="/data/twkim/diffusion/personalization/collected/images/vase";
export CUDA_VISIBLE_DEVICES=0;
accelerate launch train_custom_diffusion_single.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=./gen_reg/samples_vase/ \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --class_prompt="vase" --num_class_images=200 \
  --instance_prompt="photo of a <new1> vase"  \
  --resolution=512  \
  --train_batch_size=8  \
  --learning_rate=8e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --scale_lr --hflip  \
  --modifier_token "<new1>"