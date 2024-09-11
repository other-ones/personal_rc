backpack_dog      cat1  cat_statue  dog3  backpack_dog  pet_dog1    rc_car  teddybear
backpack_dog_dog  backpack_dog  chair1      backpack_dog  backpack_dog  poop_emoji  backpack_dog  backpack_dog

# mlm
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR1="/data/twkim/diffusion/personalization/collected/images/backpack_dog"
export CUDA_VISIBLE_DEVICES=4;
accelerate launch --main_process_port 1234  ti_key_bg_sim.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR1 \
  --resolution=512 \
  --train_batch_size=1 \
  --max_train_steps=3001 \
  --learning_rate=5e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="saved_models/analysis2/ti_plot_noprior" \
  --seed=7776 \
  --mask_tokens="[MASK]" \
  --freeze_mask_embedding=1 \
  --mlm_batch_size=400 \
  --placeholder_token1="<backpack_dog>" \
  --prior_concept1="backpack_dog" \
  --learned_embed_path1="saved_models/ti_models/single_capv8_noprior_seed2940_rep1/backpack_dog/ti_cnetv4_noprior_mlm0001_backpack_dog_mprob015_mbatch25/checkpoints/learned_embeds_s3000.pt" \
  --include_prior_concept=0 \
  --run_name='backpack_dog_mlm0001'

# nomlm
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR1="/data/twkim/diffusion/personalization/collected/images/backpack_dog"
export CUDA_VISIBLE_DEVICES=7;
accelerate launch --main_process_port 1237  ti_key_bg_sim.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir1=$DATA_DIR1 \
  --resolution=512 \
  --train_batch_size=1 \
  --max_train_steps=3001 \
  --learning_rate=5e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="saved_models/analysis2/ti_plot_noprior" \
  --seed=7776 \
  --mask_tokens="[MASK]" \
  --freeze_mask_embedding=1 \
  --mlm_batch_size=400 \
  --placeholder_token1="<backpack_dog>" \
  --prior_concept1="backpack_dog" \
  --learned_embed_path1="saved_models/ti_models/single_capv8_noprior_seed2940_rep1/backpack_dog/ti_cnetv4_noprior_nomlm_backpack_dog/checkpoints/learned_embeds_s3000.pt" \
  --include_prior_concept=0 \
  --run_name='backpack_dog_nomlm'


