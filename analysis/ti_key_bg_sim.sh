dog6      dog6  dog6  dog6  dog6  pet_dog1    dog6  dog6
dog6_dog_dog  dog6  dog6      dog6  dog6  poop_emoji  dog6  dog6




# mlm
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR1="/data/twkim/diffusion/personalization/collected/images/dog6"
export CUDA_VISIBLE_DEVICES=0;
accelerate launch --main_process_port 1230  ti_key_bg_sim.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR1 \
  --resolution=512 \
  --train_batch_size=1 \
  --max_train_steps=3001 \
  --learning_rate=5e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="saved_models/analysis_sim/ti_sim" \
  --seed=7776 \
  --mask_tokens="[MASK]" \
  --freeze_mask_embedding=1 \
  --mlm_batch_size=400 \
  --placeholder_token1="<dog6>" \
  --train_prior_concept1="cat" \
  --learned_embed_path1="saved_models/ti_models/bigger_reduced4_prior_seed7777_qlab03_rep2/dog6/ti_bigger_qlab03_prior_mlm0001_dog6_mprob015_mbatch25_mtarget_masked/checkpoints/learned_embeds_s3000.pt" \
  --mask_embed_path='saved_models/mlm_models/sd1_contextnetv6_nonpadding_1e4_unnorm_mprob015_batch150_bigger_synthcap/checkpoints/checkpoint-100000/mask_embeds_100000_ckpt.pt' \
  --include_prior_concept=0 \
  --run_name='dog6_mlm0001'

# nomlm
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR1="/data/twkim/diffusion/personalization/collected/images/dog6"
export CUDA_VISIBLE_DEVICES=1;
accelerate launch --main_process_port 1231  ti_key_bg_sim.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR1 \
  --resolution=512 \
  --train_batch_size=1 \
  --max_train_steps=3001 \
  --learning_rate=5e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="saved_models/analysis_sim/ti_sim" \
  --seed=7776 \
  --mask_tokens="[MASK]" \
  --freeze_mask_embedding=1 \
  --mlm_batch_size=400 \
  --placeholder_token1="<dog6>" \
  --train_prior_concept1="dog6" \
  --learned_embed_path1="saved_models/ti_models/bigger_reduced4_prior_seed7777_qlab03_rep2/dog6/ti_bigger_qlab03_prior_nomlm_dog6/checkpoints/learned_embeds_s3000.pt" \
  --mask_embed_path='saved_models/mlm_models/sd1_contextnetv6_nonpadding_1e4_unnorm_mprob015_batch150_bigger_synthcap/checkpoints/checkpoint-100000/mask_embeds_100000_ckpt.pt' \
  --include_prior_concept=0 \
  --run_name='dog6_nomlm'