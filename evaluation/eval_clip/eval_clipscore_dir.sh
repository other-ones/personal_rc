export CUDA_VISIBLE_DEVICES=6;
python eval_clipscore_dir.py  \
--dir_path='../../dreambooth/results/db_results/single_capv7_seed2940_rep1' \
--ignore_legacy=0

export CUDA_VISIBLE_DEVICES=6;
python eval_clipscore_dir.py  \
--dir_path='../../dreambooth/results/db_results/single_capv7_seed2940_rep1' \
--ignore_legacy=0

export CUDA_VISIBLE_DEVICES=7;
python eval_clipscore_dir.py  \
--dir_path='../../dreambooth/results/db_results/single_mtarget_seed7777_rep1' \
--ignore_legacy=0