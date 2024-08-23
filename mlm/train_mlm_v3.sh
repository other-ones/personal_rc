
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export MODEL_NAME="stabilityai/stable-diffusion-2-1";


export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=0,4;
accelerate launch --num_processes=2 --main_process_port=7354 train_mlm_v3.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir="saved_models/mlm_models" \
  --train_batch_size=128 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=100001 \
  --save_steps=10000 \
  --local_rank=0 \
  --visualize_steps=25 \
  --seed=7776 \
  --mask_tokens='[MASK]' \
  --caption_path='/data/dataset/coco/karpathy/coco_caption_raw.txt' \
  --run_name='sd1_contextnetv3_nonpadding_1e4' \
  --mlm_weight=5 \
  --mlm_target='non_padding' \
  --whole_word_mask=0 \
  --checkpoints_total_limit=10 \
  --report_to="wandb" \
  --project_name='ContextNetV3 Train' 



export MODEL_NAME="stabilityai/stable-diffusion-2-1";
export CUDA_VISIBLE_DEVICES=0,1;
accelerate launch --num_processes=2 --main_process_port=7354 train_mlm_v2.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir="saved_models/mlm_models" \
  --train_batch_size=128 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=100001 \
  --save_steps=10000 \
  --local_rank=0 \
  --visualize_steps=25 \
  --seed=7776 \
  --mask_tokens='[MASK]' \
  --caption_path='/data/dataset/coco/karpathy/coco_caption_raw.txt' \
  --run_name='sd2_contextnetv2_nonpadding_1e4' \
  --mlm_weight=5 \
  --mlm_target='non_padding' \
  --whole_word_mask=0 \
  --checkpoints_total_limit=10 \
  --report_to="wandb" \
  --project_name='ContextNetV2 Train' 




export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export MODEL_NAME="stabilityai/stable-diffusion-2-1";
export CUDA_VISIBLE_DEVICES=2,3;
accelerate launch --num_processes=2 --main_process_port=7353 train_mlm_v2.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir="saved_models/mlm_models" \
  --train_batch_size=128 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=100001 \
  --save_steps=10000 \
  --local_rank=0 \
  --visualize_steps=25 \
  --num_inference_steps=25 \
  --seed=7776 \
  --mask_tokens='[MASK]' \
  --caption_path='/data/dataset/coco/karpathy/coco_caption_raw.txt' \
  --run_name='sd2_contextnetv2_all_1e4' \
  --mlm_weight=5 \
  --mlm_target='non_padding' \
  --whole_word_mask=0 \
  --checkpoints_total_limit=10 \
  --report_to="wandb" \
  --project_name='ContextNet Train' 

