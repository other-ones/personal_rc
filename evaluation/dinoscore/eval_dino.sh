export CUDA_VISIBLE_DEVICES=1;
python eval_dino.py \
--method_root='./../../dreambooth' \
--keywords='single_seed2940' \
--grounded=0

export CUDA_VISIBLE_DEVICES=7;
python eval_all.py \
--method_root='./../../textual_inversion' \
--keywords='single_prior' \
--grounded=1 \
--sort=0