import os
import numpy as np

model_root='/home/twkim/project/rich_context/dreambooth/saved_models/db_models/single_capv7_seed2940_rep1'
concepts=os.listdir(model_root)
for concept in concepts:
    model_concept_path=os.path.join(model_root,concept)
    exps=os.listdir(model_concept_path)
    for exp in exps:
        if not '_ti_s' in exp:
            continue
        src_model_exp_path=os.path.join(model_concept_path,exp)
        exp_new=exp.replace('_ti_s','_s')
        exp_new+='_ti'
        dst_model_exp_path=os.path.join(model_concept_path,exp_new)
        if os.path.exists(src_model_exp_path):
            os.rename(src_model_exp_path,dst_model_exp_path)
            print(exp,'model renamed to',exp_new)

# results
result_root='/home/twkim/project/rich_context/dreambooth/results/db_results/single_capv7_seed2940_rep1'
concepts=os.listdir(result_root)
for concept in concepts:
    result_concept_path=os.path.join(result_root,concept)
    exps=os.listdir(result_concept_path)
    for exp in exps:
        if not '_ti_s' in exp:
            continue
        src_result_exp_path=os.path.join(result_concept_path,exp)
        exp_new=exp.replace('_ti_s','_s')
        exp_new+='_ti'
        dst_result_exp_path=os.path.join(result_concept_path,exp_new)
        if os.path.exists(src_result_exp_path):
            os.rename(src_result_exp_path,dst_result_exp_path)
            print(exp,'result renamed to',exp_new)