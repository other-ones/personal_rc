export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export OUTPUT_DIR="saved_models/custom_diffusion/single";
export INSTANCE_DIR="/data/twkim/diffusion/personalization/collected/images/pet_cat1";
export CUDA_VISIBLE_DEVICES=0;
accelerate launch generate_prior.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --class_data_dir1=./priors/samples_sunglasses/ \
  --resolution=512  \
  --class_prompt1="sunglasses" --num_class_images=200 \
  --sample_batch_size=1;


  export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export OUTPUT_DIR="saved_models/custom_diffusion/single";
export INSTANCE_DIR="/data/twkim/diffusion/personalization/collected/images/pet_cat1";
export CUDA_VISIBLE_DEVICES=1;
accelerate launch generate_prior.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --class_data_dir1=./priors/samples_barn/ \
  --resolution=512  \
  --class_prompt1="sunglasses" --num_class_images=200 \
  --sample_batch_size=20;