export CUDA_VISIBLE_DEVICES=1;
python eval_clipscore_dir.py  \
--dir_path='../../results/single' 


export CUDA_VISIBLE_DEVICES=7;
python eval_clipscore_dir.py  \
--dir_path='../../results/multi_raw' --target_keyword='nomlm'



export CUDA_VISIBLE_DEVICES=5;
python eval_clipscore_dir.py  \
--dir_path='../../textual_inversion/results/sd2/single_prior' 




export CUDA_VISIBLE_DEVICES=7;
python eval_clipscore_dir.py  \
--dir_path='../../DisenBooth/results/disenbooth/single' 

# results/disenbooth/single/dog6/disenbooth_nopp_nomlm_dog6_lr1e4_alr1e4