export CUDA_VISIBLE_DEVICES=0;
python eval_clipscore_dir_prior.py  \
--dir_path='../../custom_diffusion/results/priors/sd1' \
--ignore_legacy=0
