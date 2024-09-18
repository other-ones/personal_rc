export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUDA_VISIBLE_DEVICES=2;
accelerate launch generate_prior.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --class_data_dir1=/data/twkim/diffusion/personalization/priors/samples_teddy/ \
  --resolution=512  \
  --class_prompt1="teddy bear" --num_class_images=200 \
  --sample_batch_size=20


export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUDA_VISIBLE_DEVICES=5;
accelerate launch generate_prior.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --class_data_dir1=/data/twkim/diffusion/personalization/priors/samples_teapot/ \
  --resolution=512  \
  --class_prompt1="teapot" --num_class_images=200 \
  --sample_batch_size=20;



  export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export OUTPUT_DIR="saved_models/custom_diffusion/single";
export INSTANCE_DIR="/data/twkim/diffusion/personalization/collected/images/pet_cat1";
export CUDA_VISIBLE_DEVICES=3;
accelerate launch generate_prior.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --class_data_dir1=./priors/samples_barn/ \
  --resolution=512  \
  --class_prompt1="barn" --num_class_images=200 \
  --sample_batch_size=20;


export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export OUTPUT_DIR="saved_models/custom_diffusion/single";
export INSTANCE_DIR="/data/twkim/diffusion/personalization/collected/images/pet_cat1";
export CUDA_VISIBLE_DEVICES=2;
accelerate launch generate_prior.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --class_data_dir1=./priors/samples_statue/ \
  --resolution=512  \
  --class_prompt1="statue" --num_class_images=200 \
  --sample_batch_size=20;


export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export OUTPUT_DIR="saved_models/custom_diffusion/single";
export INSTANCE_DIR="/data/twkim/diffusion/personalization/collected/images/pet_cat1";
export CUDA_VISIBLE_DEVICES=5;
accelerate launch generate_prior.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --class_data_dir1=./priors/samples_backpack/ \
  --resolution=512  \
  --class_prompt1="backpack" --num_class_images=200 \
  --sample_batch_size=20;


  export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export OUTPUT_DIR="saved_models/custom_diffusion/single";
export INSTANCE_DIR="/data/twkim/diffusion/personalization/collected/images/pet_cat1";
export CUDA_VISIBLE_DEVICES=6;
accelerate launch generate_prior.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --class_data_dir1=./priors/samples_chair/ \
  --resolution=512  \
  --class_prompt1="chair" --num_class_images=200 \
  --sample_batch_size=20;


  
  export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export OUTPUT_DIR="saved_models/custom_diffusion/single";
export INSTANCE_DIR="/data/twkim/diffusion/personalization/collected/images/pet_cat1";
export CUDA_VISIBLE_DEVICES=7;
accelerate launch generate_prior.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --class_data_dir1=./priors/samples_rc_car/ \
  --resolution=512  \
  --class_prompt1="rc car" --num_class_images=200 \
  --sample_batch_size=20;

export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export OUTPUT_DIR="saved_models/custom_diffusion/single";
export INSTANCE_DIR="/data/twkim/diffusion/personalization/collected/images/pet_cat1";
export CUDA_VISIBLE_DEVICES=1;
accelerate launch generate_prior.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --class_data_dir1=./priors/samples_flower/ \
  --resolution=512  \
  --class_prompt1="flower" --num_class_images=200 \
  --sample_batch_size=20;

  export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export OUTPUT_DIR="saved_models/custom_diffusion/single";
export INSTANCE_DIR="/data/twkim/diffusion/personalization/collected/images/pet_cat1";
export CUDA_VISIBLE_DEVICES=4;
accelerate launch generate_prior.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --class_data_dir1=./priors/samples_teddybear/ \
  --resolution=512  \
  --class_prompt1="teddy bear" --num_class_images=200 \
  --sample_batch_size=20;