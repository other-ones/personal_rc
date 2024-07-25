export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR1="/data/twkim/diffusion/personalization/collected/images/pet_cat1"
export DATA_DIR2="/data/twkim/diffusion/personalization/collected/images/pet_dog1"
export CUDA_VISIBLE_DEVICES=2;
accelerate launch --main_process_port 4232  train_dreambooth_multi_mlm.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR1 \
  --train_data_dir2=$DATA_DIR2 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=2001 \
  --learning_rate=1e-6 \
  --lr_warmup_steps=0 \
  --output_dir="saved_models/dreambooth_models/multi/" \
  --mask_tokens="[MASK]" \
  --lambda_mlm=0.001 --freeze_mask_embedding=1 \
  --cls_net_path='saved_models/mlm_contextnet_nonpad_lr1e4/checkpoints/cls_net_99000_ckpt.pt' \
  --mask_embed_path='saved_models/mlm_contextnet_nonpad_lr1e4/checkpoints/mask_embeds_99000_ckpt.pt' \
  --mlm_target='masked' \
  --mlm_batch_size=20 \
  --run_name='pet_cat1_pet_dog1_multi_mlm0001_sim0_scratch' \
  --prompt_type='two_pets' \
  --include_prior_concept=1 \
  --train_text_encoder \
  --validation_steps=100 \
  --with_prior_preservation=1 \
  --placeholder_token1="<pet_cat1>" \
  --class_prompt1="a picture of a cat" \
  --class_data_dir1="priors/cat" \
  --prior_concept1="cat" \
  --placeholder_token2="<pet_dog1>" \
  --class_prompt2="a picture of a dog" \
  --class_data_dir2="priors/dog" \
  --prior_concept2="dog" \
  --simple_caption=1 \
  --make_composition=1 \
  --masked_loss=1


export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR1="/data/twkim/diffusion/personalization/collected/images/pet_cat1"
export DATA_DIR2="/data/twkim/diffusion/personalization/collected/images/pet_dog1"
export CUDA_VISIBLE_DEVICES=3;
accelerate launch --main_process_port 5123  train_dreambooth_multi_mlm.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir2=$DATA_DIR2 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=2001 \
  --learning_rate=1e-6 \
  --lr_warmup_steps=0 \
  --output_dir="saved_models/dreambooth_models/multi/" \
  --mask_tokens="[MASK]" \
  --lambda_mlm=0.001 --freeze_mask_embedding=1 \
  --cls_net_path='saved_models/mlm_contextnet_nonpad_lr1e4/checkpoints/cls_net_99000_ckpt.pt' \
  --mask_embed_path='saved_models/mlm_contextnet_nonpad_lr1e4/checkpoints/mask_embeds_99000_ckpt.pt' \
  --mlm_target='masked' \
  --mlm_batch_size=20 \
  --run_name='pet_cat1_pet_dog1_multi_mlm0001_sim0_nomask_nocompose_scratch' \
  --prompt_type='two_pets' \
  --include_prior_concept=1 \
  --train_text_encoder \
  --validation_steps=100 \
  --with_prior_preservation=1 \
  --placeholder_token1="<pet_cat1>" \
  --train_data_dir1=$DATA_DIR1 \
  --class_prompt1="a picture of a cat" \
  --class_data_dir1="priors/cat" \
  --prior_concept1="cat" \
  --placeholder_token2="<pet_dog1>" \
  --class_prompt2="a picture of a dog" \
  --class_data_dir2="priors/dog" \
  --prior_concept2="dog" \
  --simple_caption=1 \
  --make_composition=0 \
  --masked_loss=0 