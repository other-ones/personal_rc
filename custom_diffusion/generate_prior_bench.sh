export MODEL_NAME="runwayml/stable-diffusion-v1-5";
export CUDA_VISIBLE_DEVICES=2;
accelerate launch generate_prior_bench.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --resolution=512  \
  --eval_batch_size=15 \
  --benchmark_path='../datasets_pkgs/eval_prompts/dreambooth.json'

