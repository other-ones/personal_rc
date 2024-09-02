import os
import shutil
root='results/single/'
concepts=os.listdir(root)
for concept in concepts:
    concept_path=os.path.join(root,concept)
    exps=os.listdir(concept_path)
    for exp in exps:
        exp_path=os.path.join(concept_path,exp)
        if 'masked' in exp:
            shutil.rmtree(exp_path)