# ti
results/ti_results/bigger_reduced4_prior_seed7777_qlab03_rep2

python print_results_all.py \
--dir_path=results/ti_results/bigger_reduced4_prior_seed7777_qlab03_rep2 
# pplus
results/pplus_results/bigger_reduced4_prior_seed7777_qlab03_rep1

python print_results_all.py \
--dir_path=results/pplus_results/bigger_reduced4_prior_seed7777_qlab03_rep1 

# db
results/db_results/bigger_seed7777_qlab03_rep1
results/db_results/bigger_seed7777_qlab03_rep2
python print_results_all.py \
--dir_path=results/db_results/bigger_seed7777_qlab03_rep2 

# cd
results/cd_results/highmlm_seed2940_qlab03_rep1
results/cd_results/init_seed2940_qlab03_rep2_no_ti
python print_results_all.py \
--dir_path=results/cd_results/highmlm_seed2940_qlab03_rep1 
python print_results_all.py \
--dir_path=results/cd_results/init_seed2940_qlab03_rep2_no_ti 

# ablation
results/ti_results/abl_mprob_prior_seed7777_qlab03_rep1
results/ti_results/abl_prompt_size_prior_seed7777_qlab03_rep1

