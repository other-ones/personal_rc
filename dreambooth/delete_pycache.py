
import os
import shutil



# root='/home/twkim/project/rich_context/dreambooth/saved_models/ti_models/singlev3_prior_seed2940_v1'
root='/home/twkim/project/rich_context/dreambooth/saved_models/db_models'
dirs=os.listdir(root)
for dir in dirs:
    dir_path=os.path.join(root,dir)
    concepts=os.listdir(dir_path)
    for concept in concepts:
        concept_path=os.path.join(dir_path,concept)
        exps=os.listdir(concept_path)
        for exp in exps:
            exp_path=os.path.join(concept_path,exp)
            cache_path=os.path.join(exp_path,'src/packages/diffusers/pipelines/stable_diffusion/__pycache__')
            print(os.path.exists(cache_path))
            if os.path.exists(cache_path):
                shutil.rmtree(cache_path)
                print(exp)
