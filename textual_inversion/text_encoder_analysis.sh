export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR1="/data/twkim/diffusion/personalization/collected/images/pet_cat1"
export DATA_DIR2="/data/twkim/diffusion/personalization/collected/images/pet_dog1"
export CUDA_VISIBLE_DEVICES=0;
accelerate launch --main_process_port 1234  text_encoder_analysis.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR1 \
  --train_data_dir2=$DATA_DIR2 \
  --learnable_property="object" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3001 \
  --learning_rate=5e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="saved_models/analysis" \
  --seed=7776 \
  --mask_tokens="[MASK]" \
  --lambda_mlm=0.001 --freeze_mask_embedding=1 --lambda_contrastive=0 \
  --cls_net_path='saved_models/mlm_contextnet_nonpad_lr1e4/checkpoints/cls_net_99000_ckpt.pt' \
  --mask_embed_path='saved_models/mlm_contextnet_nonpad_lr1e4/checkpoints/mask_embeds_99000_ckpt.pt' \
  --mlm_target='masked' \
  --mlm_batch_size=400 \
  --prompt_type='two_pets' \
  --placeholder_token1="<pet_cat1>" \
  --placeholder_token2="<pet_dog1>" \
  --prior_concept1="cat" \
  --prior_concept2="dog" \
  --learned_embed_path1="saved_models/single/pet_cat1/ti_noprior_mlm0001_pet_cat1/checkpoints/learned_embeds_s3000.pt" \
  --learned_embed_path2="saved_models/single/pet_dog1/ti_noprior_mlm0001_pet_dog1/checkpoints/learned_embeds_s3000.pt" \
  --include_prior_concept=0 \
  --scale_lr \
  --run_name='tmp' \
  --make_composition=1 \
  --masked_loss=1 


  