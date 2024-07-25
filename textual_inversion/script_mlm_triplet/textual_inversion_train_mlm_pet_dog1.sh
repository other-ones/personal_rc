export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/pet_dog1"
export CUDA_VISIBLE_DEVICES=0;
accelerate launch  textual_inversion_train_mlm_triplet.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3001 \
  --learning_rate=5e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="saved_models/pet_dog1" \
  --seed=7777 \
  --mask_tokens="[MASK]" \
  --lambda_mlm=0.0005 --freeze_mask_embedding=1 --lambda_triplet=0.01 \
  --cls_net_path='saved_models/mlm_contextnet_nonpad_lr1e4/checkpoints/cls_net_99000_ckpt.pt' \
  --mask_embed_path='saved_models/mlm_contextnet_nonpad_lr1e4/checkpoints/mask_embeds_99000_ckpt.pt' \
  --mlm_target='masked' \
  --mlm_batch_size=50 \
  --run_name='tiv2_mlm00005_triplet001_m01_pet_dog1' \
  --scale_lr \
  --placeholder_token="<pet_dog1>" \
  --initializer_token="dog" \
  --prompt_type='pet' \
  --prior_concept="dog" \
  --include_prior_concept=1 \
  --margin=0.1 \
  --report_to='wandb' \
  --project_name='TI MLM V3 Triplet' 

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/pet_dog1"
export CUDA_VISIBLE_DEVICES=1;
accelerate launch  textual_inversion_train_mlm_triplet.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3001 \
  --learning_rate=5e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="saved_models/pet_dog1" \
  --seed=7777 \
  --mask_tokens="[MASK]" \
  --lambda_mlm=0.0005 --freeze_mask_embedding=1 --lambda_triplet=0.01 \
  --cls_net_path='saved_models/mlm_contextnet_nonpad_lr1e4/checkpoints/cls_net_99000_ckpt.pt' \
  --mask_embed_path='saved_models/mlm_contextnet_nonpad_lr1e4/checkpoints/mask_embeds_99000_ckpt.pt' \
  --mlm_target='masked' \
  --mlm_batch_size=50 \
  --run_name='tiv2_mlm00005_triplet001_m03_pet_dog1' \
  --scale_lr \
  --placeholder_token="<pet_dog1>" \
  --initializer_token="dog" \
  --prompt_type='pet' \
  --prior_concept="dog" \
  --include_prior_concept=1 \
  --margin=0.3 \
  --report_to='wandb' \
  --project_name='TI MLM V3 Triplet' 


export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="/data/twkim/diffusion/personalization/collected/images/pet_dog1"
export CUDA_VISIBLE_DEVICES=2;
accelerate launch  textual_inversion_train_mlm_triplet.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3001 \
  --learning_rate=5e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="saved_models/pet_dog1" \
  --seed=7777 \
  --mask_tokens="[MASK]" \
  --lambda_mlm=0.0005 --freeze_mask_embedding=1 --lambda_triplet=0.01 \
  --cls_net_path='saved_models/mlm_contextnet_nonpad_lr1e4/checkpoints/cls_net_99000_ckpt.pt' \
  --mask_embed_path='saved_models/mlm_contextnet_nonpad_lr1e4/checkpoints/mask_embeds_99000_ckpt.pt' \
  --mlm_target='masked' \
  --mlm_batch_size=50 \
  --run_name='tiv2_mlm00005_triplet001_m05_pet_dog1' \
  --scale_lr \
  --placeholder_token="<pet_dog1>" \
  --initializer_token="dog" \
  --prompt_type='pet' \
  --prior_concept="dog" \
  --include_prior_concept=1 \
  --margin=0.5 \
  --report_to='wandb' \
  --project_name='TI MLM V3 Triplet' 
