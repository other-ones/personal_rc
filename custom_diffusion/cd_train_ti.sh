export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export OUTPUT_DIR="saved_models/cd_models/single/pet_cat1";
export CUBLAS_WORKSPACE_CONFIG=:4096:8;
export INSTANCE_DIR="/data/twkim/diffusion/personalization/collected/images/pet_cat1";
export CUDA_VISIBLE_DEVICES=2;
accelerate launch --main_process_port=9980 cd_train_clean.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_data_dir1=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --resolution=512  \
  --train_batch_size=2  \
  --learning_rate=1e-3  \
  --lr_warmup_steps=0 \
  --max_train_steps=501 \
  --scale_lr  \
  --with_prior_preservation=1 \
  --prior_loss_weight=1.0 \
  --num_class_images=200 \
  --placeholder_token1="<pet_cat1>" \
  --mask_tokens="[MASK]" \
  --class_data_dir1=./priors/samples_dog/ \
  --lambda_mlm=0.001 \
  --class_prompt1="a picture of a dog" \
  --train_prompt_type="pet" \
  --eval_prompt_type="living" \
  --run_name="tmp" \
  --mlm_target="masked" \
  --validation_steps=100 \
  --train_text_encoder=0 \
  --train_prior_concept1="dog" \
  --eval_prior_concept1="dog" \
  --caption_root='../datasets_pkgs/captions/v7' \
  --seed=2940 \
  --mlm_batch_size=25 \
  --initializer_token="sks" \
  --mask_prob=0.25 \
  --cls_net_path='saved_models/mlm_models/sd1_contextnetv4_nonpadding_1e4_unnorm_mprob015_batch150/checkpoints/checkpoint-100000/cls_net_100000_ckpt.pt' \
  --mask_embed_path='saved_models/mlm_models/sd1_contextnetv4_nonpadding_1e4_unnorm_mprob015_batch150/checkpoints/checkpoint-100000/mask_embeds_100000_ckpt.pt' \
  --resume_cd_path='saved_models/cd_models/bigger_seed7777_qlab07_rep1/pet_cat1/cd_bigger_qlab07_mlm0001_pet_cat1_mprob025_mbatch25_mtarget_masked_lr1e5/checkpoints/checkpoint-500/custom_diffusion.pt' \
  --learned_embed_path1='saved_models/cd_models/bigger_seed7777_qlab07_rep1/pet_cat1/cd_bigger_qlab07_mlm0001_pet_cat1_mprob025_mbatch25_mtarget_masked_lr1e5/checkpoints/checkpoint-500/learned_embeds.pt'