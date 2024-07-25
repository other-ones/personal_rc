
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export MODEL_NAME="stabilityai/stable-diffusion-2-1";
export CUDA_VISIBLE_DEVICES=2,3;
accelerate launch --num_processes=2 --main_process_port=7354 train_mlm.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir="saved_models" \
  --train_batch_size=150 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=100000 \
  --save_steps=1000 \
  --local_rank=0 \
  --visualize_steps=25 \
  --num_inference_steps=25 \
  --seed=7776 \
  --mask_tokens='[MASK]' \
  --caption_path='/data/dataset/coco/karpathy/coco_caption_raw.txt' \
  --run_name='mlm_contextnet_nonpad_lr1e4' \
  --mlm_weight=5 \
  --mlm_target='non_padding' \
  --whole_word_mask=0 \
  --project_name='SD MLM' 




# stabilityai/stable-diffusion-2-1
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export MODEL_NAME="stabilityai/stable-diffusion-2-1";
export CUDA_VISIBLE_DEVICES=0;
accelerate launch --main_process_port=2940 train_mlm.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir="saved_models" \
  --train_batch_size=150 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-3 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=100000 \
  --save_steps=1000 \
  --local_rank=0 \
  --visualize_steps=25 \
  --num_inference_steps=25 \
  --seed=7776 \
  --mask_tokens='[MASK]' \
  --caption_path='/data/dataset/bookcorpus/corpus_sampled.txt' \
  --report_to='wandb' \
  --project_name='SD MLM' \
  --mlm_target='all' \
  --run_name='tmp' \
  --mlm_weight=2 


# stabilityai/stable-diffusion-2-1
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=0;
accelerate launch --main_process_port=4344 train_mlm.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir="saved_models" \
  --train_batch_size=150 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-3 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=100000 \
  --save_steps=1000 \
  --local_rank=0 \
  --visualize_steps=25 \
  --num_inference_steps=25 \
  --seed=7776 \
  --mask_tokens='[MASK]' \
  --caption_path='/data/dataset/bookcorpus/corpus_sampled2.txt' \
  --project_name='SD MLM' \
  --report_to='wandb' \
  --run_name='tmp' \
  --mlm_weight=2 \
  --mlm_target='all' \
  --whole_word_mask=0


# stabilityai/stable-diffusion-2-1
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=0,1;
accelerate launch --num_processes=2 --main_process_port=7344 train_mlm.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir="saved_models" \
  --train_batch_size=150 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=100000 \
  --save_steps=1000 \
  --local_rank=0 \
  --visualize_steps=25 \
  --num_inference_steps=25 \
  --seed=7776 \
  --mask_tokens='[MASK]' \
  --caption_path='/data/dataset/coco/karpathy/coco_caption_raw.txt' \
  --project_name='SD MLM' \
  --run_name='mlm_contextnet_all_lr1e4' \
  --mlm_weight=5 \
  --mlm_target='all' \
  --report_to='wandb' \
  --whole_word_mask=0

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=2,3;
accelerate launch --num_processes=2 --main_process_port=7354 train_mlm.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir="saved_models" \
  --train_batch_size=150 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=100000 \
  --save_steps=1000 \
  --local_rank=0 \
  --visualize_steps=25 \
  --num_inference_steps=25 \
  --seed=7776 \
  --mask_tokens='[MASK]' \
  --caption_path='/data/dataset/coco/karpathy/coco_caption_raw.txt' \
  --project_name='SD MLM' \
  --run_name='mlm_contextnet_nonpad_lr1e4' \
  --mlm_weight=5 \
  --mlm_target='non_padding' \
  --report_to='wandb' \
  --whole_word_mask=0


export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CUDA_VISIBLE_DEVICES=4,5;
accelerate launch --num_processes=2 --main_process_port=9853 train_mlm.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir="saved_models" \
  --train_batch_size=150 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-3 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=100000 \
  --save_steps=1000 \
  --local_rank=0 \
  --visualize_steps=25 \
  --num_inference_steps=25 \
  --seed=7776 \
  --mask_tokens='[MASK]' \
  --caption_path='/data/dataset/coco/karpathy/coco_caption_raw.txt' \
  --project_name='SD MLM' \
  --run_name='mlm_contextnet_masked' \
  --mlm_weight=5 \
  --mlm_target='masked' \
  --report_to='wandb' \
  --whole_word_mask=0