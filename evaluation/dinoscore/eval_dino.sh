export CUDA_VISIBLE_DEVICES=1;
python eval_dino.py \
--method_root='./../../dreambooth' \
--keywords='singlev2_seed2940' \
--grounded=0

export CUDA_VISIBLE_DEVICES=1;
python eval_dino.py \
--method_root='./../../textual_inversion' \
--keywords='singlev3_prior_seed2940' \
--grounded=0  --exclude_attr_change=1