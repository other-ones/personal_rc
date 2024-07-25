export CUDA_VISIBLE_DEVICES=1;
python eval_clipscore_dir.py  \
--dir_path='../../results/single' 


export CUDA_VISIBLE_DEVICES=7;
python eval_clipscore_dir.py  \
--dir_path='../../results/multi_raw' --target_keyword='nomlm'



export CUDA_VISIBLE_DEVICES=1;
python eval_clipscore_dir.py  \
--dir_path='../../pplus/results/pplus_results/single' 

