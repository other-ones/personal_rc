import os
import json
import numpy as np
root='results/ti_attns_seed7777/bigger_reduced4_prior_seed7777_qlab03_rep2/'
concepts=os.listdir(root)
nomlm_list=[]
mlm_list=[]
for concept in concepts:
    concept_path=os.path.join(root,concept)
    exps=os.listdir(concept_path)
    for exp in exps:
        exp_path=os.path.join(concept_path,exp)
        log_path=os.path.join(exp_path,'log.out')
        lines=open(log_path).readlines()
        for idx, line in enumerate(lines):
            line=line.strip()
            if '19 avg_ce' in line:
                print(idx,line)
                ce=float(line.split()[-1])
        if '_nomlm_' in exp:
            nomlm_list.append(ce)
        else:
            mlm_list.append(ce)
print(f'nomlm\t{np.mean(nomlm_list)}')
print(f'mlm\t{np.mean(mlm_list)}')
