import os
import numpy as np

src_root='/home/twkim/project/rich_context/dreambooth/saved_models/db_models/single_capv7_ti_seed2940_rep1'
dst_root='/home/twkim/project/rich_context/dreambooth/saved_models/db_models/single_capv7_seed2940_rep1'
concepts=os.listdir(src_root)
for concept in concepts:
    src_concept_path=os.path.join(src_root,concept)
    dst_concept_path=os.path.join(dst_root,concept)
    exps=os.listdir(src_concept_path)
    for exp in exps:
        src_exp_path=os.path.join(src_concept_path,exp)
        dst_exp_path=os.path.join(dst_concept_path,exp)
        if os.path.exists(src_exp_path):
            os.rename(src_exp_path,dst_exp_path)
            print(exp,'moved')

# results
src_root='/home/twkim/project/rich_context/dreambooth/results/db_results/single_capv7_ti_seed2940_rep1'
dst_root='/home/twkim/project/rich_context/dreambooth/results/db_results/single_capv7_seed2940_rep1'
concepts=os.listdir(src_root)
for concept in concepts:
    src_concept_path=os.path.join(src_root,concept)
    dst_concept_path=os.path.join(dst_root,concept)
    exps=os.listdir(src_concept_path)
    # print(os.path.exists(dst_concept_path))
    for exp in exps:
        src_exp_path=os.path.join(src_concept_path,exp)
        dst_exp_path=os.path.join(dst_concept_path,exp)
        print(src_exp_path,os.path.exists(src_exp_path))
        if os.path.exists(src_exp_path):
            os.rename(src_exp_path,dst_exp_path)
            print(exp,'moved')
        # print(exp,'results moved')
        