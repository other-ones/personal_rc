export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export OUTPUT_DIR="outputs/pet_cat1_wooden_pot";
export INSTANCE_DIR="/data/twkim/diffusion/personalization/collected/images/pet_cat1";
export CUDA_VISIBLE_DEVICES=0,1;
  accelerate launch --main_process_port=9982 train_custom_diffusion_origin.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir=./logs/cat_wooden_pot  \
  --concepts_list=./assets/concept_list.json --prior_loss_weight=1.0 \
  --resolution=512  \
  --train_batch_size=4  \
  --learning_rate=1e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --num_class_images=200 \
  --scale_lr --hflip  \
  --validation_prompt="The <new1> cat is sitting inside a <new2> wooden pot and looking up" \
  --validation_steps=50 \
  --modifier_token "<new1>+<new2>" \
  --with_prior_preservation
