export CUDA_VISIBLE_DEVICES=6;
python eval_clipscore_dir.py  \
--dir_path='../../dreambooth/results/db_results/single_capv7_seed2940_rep1' \
--ignore_legacy=0

export CUDA_VISIBLE_DEVICES=6;
python eval_clipscore_dir.py  \
--dir_path='../../dreambooth/results/db_results/single_capv7_seed2940_rep1' \
--ignore_legacy=0

export CUDA_VISIBLE_DEVICES=7;
python eval_clipscore_single.py  \
--dir_path='../../textual_inversion/results/ti_results/custom/racket1/nomlm/' \
--ignore_legacy=0


export CUDA_VISIBLE_DEVICES=7;
python eval_clipscore_single.py  \
--dir_path='../../textual_inversion/results/ti_results/custom/racket1/mlm0001_specific/' \
--ignore_legacy=0



export CUDA_VISIBLE_DEVICES=7;
python eval_clipscore_dir.py  \
--dir_path='/data/twkim/diffusion/personalization/custom_images/complex/racket1' \
--ignore_legacy=0