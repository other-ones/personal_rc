export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUDA_VISIBLE_DEVICES=7;
python test_custom_diffusion.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--resume_path='saved_models/custom_diffusion/single/vase/custom_nomlm_vase/checkpoints/checkpoint-250' \
--placeholder_token1="<vase>" \
--prior_concept1="vase"


export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUDA_VISIBLE_DEVICES=7;
python test_custom_diffusion.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--resume_path='saved_models/custom_diffusion/single/dog6/custom_nomlm_dog6/checkpoints/checkpoint-250' \
--placeholder_token1="<dog6>" \
--prior_concept1="dog"