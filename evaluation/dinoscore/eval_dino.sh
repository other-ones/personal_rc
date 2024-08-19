export CUDA_VISIBLE_DEVICES=1;
python eval_dino.py \
--method_root='./../../custom_diffusion' \
--keywords='single_seed7777' \
--grounded=0

export CUDA_VISIBLE_DEVICES=7;
python eval_all.py \
--method_root='./../../textual_inversion' \
--keywords='single_prior' \
--grounded=1 \
--sort=0