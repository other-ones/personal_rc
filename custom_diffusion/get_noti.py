import os
src='/home/twkim/project/rich_context/custom_diffusion/results/cd_results/init_seed2940_qlab03_rep2'
dst='/home/twkim/project/rich_context/custom_diffusion/results/cd_results/init_seed2940_qlab03_rep2_no_ti'
os.makedirs(dst,exist_ok=True)
concepts=os.listdir(src)
for concept in concepts:
    src_concept_path=os.path.join(src,concept)
    exps=os.listdir(src_concept_path)
    dst_concept_path=os.path.join(dst,concept)
    os.makedirs(dst_concept_path,exist_ok=True)
    for exp in exps:
        if '_ti' in exp:
            continue
        if concept=='teddybear':
            dst_exp=exp.replace('highmlm','init')
        else:
            dst_exp=exp
        src_exp_path=os.path.join(src_concept_path,exp)
        dst_exp_path=os.path.join(dst_concept_path,dst_exp)
        if not os.path.exists(dst_exp_path):
            os.symlink(src_exp_path,dst_exp_path)
            print(exp)
