export CUDA_VISIBLE_DEVICES=0;
python eval_clipscore_dir.py  \
--dir_path='../../results/single' --target_keyword='nomlm'


export CUDA_VISIBLE_DEVICES=7;
python eval_clipscore_dir.py  \
--dir_path='../../results/multi_raw' --target_keyword='nomlm'