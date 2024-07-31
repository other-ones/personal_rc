export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export OUTPUT_DIR="outputs";
export INSTANCE_DIR="./data/cat"
export CUDA_VISIBLE_DEVICES=7;
accelerate launch train_custom_diffusion.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=./gen_reg/samples_cat/ \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --class_prompt="cat" --num_class_images=200 \
  --instance_prompt="photo of a <new1> cat"  \
  --resolution=512  \
  --train_batch_size=8  \
  --learning_rate=8e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --scale_lr --hflip  \
  --modifier_token "<new1>"
  # --real_prior acc


  export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export OUTPUT_DIR="outputs";
export INSTANCE_DIR="./data/dog"
export CUDA_VISIBLE_DEVICES=6;
accelerate launch train_custom_diffusion.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=./gen_reg/samples_dog/ \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --class_prompt="dog" --num_class_images=200 \
  --instance_prompt="photo of a <new1> dog"  \
  --resolution=512  \
  --train_batch_size=8  \
  --learning_rate=8e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --scale_lr --hflip  \
  --modifier_token "<new1>"