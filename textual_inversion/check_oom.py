import os
import numpy as np
root='results/ti_results/bigger_reduced4_prior_seed7777_qlab03_rep2'
concepts=os.listdir(root)
for concept in concepts:
    concept_path=os.path.join(root,concept)
    exps=os.listdir(concept_path)
    for exp in exps:
        exp_path=os.path.join(concept_path,exp)
        log_path=os.path.join(exp_path,'mask_log.out')
        if not os.path.exists(log_path):
            print(exp_path)
        lines=open(log_path).readlines()
        for line in lines:
            if 'OutOfMemoryError' in line or 'CUDA out of memory' in line:
                print(exp_path,'oom') 