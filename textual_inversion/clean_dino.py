import shutil
import os
root='results/debug_prior_seed2940'
concepts=os.listdir(root)
for concept in concepts:
    concept_path=os.path.join(root,concept)
    exps=os.listdir(concept_path)
    for exp in exps:
        exp_path=os.path.join(concept_path,exp)
        # dino_path=os.path.join(exp_path,'dino.json')
        masked_dino_path=os.path.join(exp_path,'masked_dino.json')
        # if os.path.exists(dino_path):
        #     print(dino_path)
        #     os.remove(dino_path)
        if os.path.exists(masked_dino_path):
            print(masked_dino_path)
            os.remove(masked_dino_path)