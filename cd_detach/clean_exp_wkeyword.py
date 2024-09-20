import shutil
import os
keyword='mprob05'
root='saved_models/custom_diffusion/single_re'
concepts=os.listdir(root)
for concept in concepts:
    concept_path=os.path.join(root,concept)
    exps=os.listdir(concept_path)
    for exp in exps:
        exp_path=os.path.join(concept_path,exp)
        if keyword in exp:
            print(exp)
            shutil.rmtree(exp_path)
        