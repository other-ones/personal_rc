import shutil
import os
import numpy as np
root='../saved_models/db_models/single_capv7_seed2940_rep1/'
concepts=os.listdir(root)
for concept in concepts:
    concept_path=os.path.join(root,concept)
    exps=os.listdir(concept_path)
    for exp in exps:
        if not '_ti' in exp:
            continue
        exp_path=os.path.join(concept_path,exp)
        if os.path.exists(exp_path):
            # shutil.rmtree(exp_path)
            print(exp_path,'removed')




