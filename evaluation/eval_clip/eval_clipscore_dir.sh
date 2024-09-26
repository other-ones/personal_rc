# ti

export CUDA_VISIBLE_DEVICES=6;
python eval_clipscore_dir.py  \
--dir_path='results/ti_results/bigger_reduced4_prior_seed7777_qlab03_rep2' \
--ignore_legacy=0


# pplus
export CUDA_VISIBLE_DEVICES=6;
python eval_clipscore_dir.py  \
--dir_path='results/pplus_results/bigger_reduced4_prior_seed7777_qlab03_rep1' \
--ignore_legacy=0

# db
export CUDA_VISIBLE_DEVICES=6;
python eval_clipscore_dir.py  \
--dir_path='results/db_results/bigger_seed7777_qlab03_rep1' \
--ignore_legacy=0
export CUDA_VISIBLE_DEVICES=7;
python eval_clipscore_dir.py  \
--dir_path='results/db_results/bigger_seed7777_qlab03_rep2' \
--ignore_legacy=0

# cd
export CUDA_VISIBLE_DEVICES=7;
python eval_clipscore_dir.py  \
--dir_path='results/cd_results/highmlm_seed2940_qlab03_rep1' \
--ignore_legacy=0
export CUDA_VISIBLE_DEVICES=6;
python eval_clipscore_dir.py  \
--dir_path='results/cd_results/init_seed2940_qlab03_rep2_no_ti' \
--ignore_legacy=0


# ablation
export CUDA_VISIBLE_DEVICES=3;
python eval_clipscore_dir.py  \
--dir_path='results/ti_results/abl_mprob_prior_seed7777_qlab03_rep1' \
--ignore_legacy=0
export CUDA_VISIBLE_DEVICES=6;
python eval_clipscore_dir.py  \
--dir_path='results/ti_results/abl_prompt_size_prior_seed7777_qlab03_rep1' \
--ignore_legacy=0

