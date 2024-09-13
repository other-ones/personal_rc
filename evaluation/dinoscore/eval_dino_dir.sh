export CUDA_VISIBLE_DEVICES=1;
python eval_dino_dir.py \
--dir_path='./../../dreambooth/results/db_results/single_capv7_seed2940_rep1' \
--grounded=1 --ignore_legacy=1

export CUDA_VISIBLE_DEVICES=7;
python eval_dino_dir.py \
--dir_path='./../../dreambooth/results/db_results/single_noacc_seed7777_rep1' \
--grounded=0 --ignore_legacy=0
