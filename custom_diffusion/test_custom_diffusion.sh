export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUDA_VISIBLE_DEVICES=7;
python test_custom_diffusion.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--resume_path='outputs/pet_cat1/checkpoints/checkpoint-250' \
--modifier_token="<new1>"