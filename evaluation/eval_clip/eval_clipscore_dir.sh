export CUDA_VISIBLE_DEVICES=1;
python eval_clipscore_dir.py  \
--dir_path='../../custom_diffusion/results/sgpu_seed7777'

export CUDA_VISIBLE_DEVICES=2;
python eval_clipscore_dir.py  \
--dir_path='../../custom_diffusion/results/sgpu_seed8881'

export CUDA_VISIBLE_DEVICES=3;
python eval_clipscore_dir.py  \
--dir_path='../../custom_diffusion/results/sgpu_seed2940'

export CUDA_VISIBLE_DEVICES=4;
python eval_clipscore_dir.py  \
--dir_path='../../custom_diffusion/results/sgpu_seed1234'






export CUDA_VISIBLE_DEVICES=5;
python eval_clipscore_dir.py  \
--dir_path='../../textual_inversion/results/sd2/single_prior' 




export CUDA_VISIBLE_DEVICES=1;
python eval_clipscore_dir.py  \
--dir_path='../../custom_diffusion/results/single_seed7777' \
--num_samples=100

# --target_keyword='vase' 
# results/disenbooth/single/dog6/disenbooth_nopp_nomlm_dog6_lr1e4_alr1e4