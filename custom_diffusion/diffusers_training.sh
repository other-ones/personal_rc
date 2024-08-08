
# export MODEL_NAME="CompVis/stable-diffusion-v1-4";
export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export INSTANCE_DIR="/data/twkim/diffusion/personalization/collected/images/pet_cat1";
export CUDA_VISIBLE_DEVICES=0,1;
accelerate launch --main_process_port=9982 diffusers_training.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR  \
    --class_data_dir=./priors/samples_cat/ \
    --output_dir=./logs/cat  \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="photo of a <new1> cat"  \
    --class_prompt="cat" \
    --resolution=512  \
    --train_batch_size=2  \
    --learning_rate=1e-5  \
    --lr_warmup_steps=0 \
    --max_train_steps=250 \
    --num_class_images=200 \
    --scale_lr --hflip  \
    --modifier_token "<new1>" \
    --enable_xformers_memory_efficient_attention 