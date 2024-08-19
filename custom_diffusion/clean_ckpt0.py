import shutil
import os
root='saved_models/custom_diffusion/single_seed7777'
concepts=os.listdir(root)
for concept in concepts:
    concept_path=os.path.join(root,concept)
    log_path=os.path.join(concept_path,'logs')
    if os.path.exists(log_path):
        print(log_path)
        shutil.rmtree(log_path)
        
    exps=os.listdir(concept_path)
    for exp in exps:
        exp_path=os.path.join(concept_path,exp)
        ckpt_path=os.path.join(exp_path,'checkpoints/checkpoint-0')
        if os.path.exists(ckpt_path):
            print(ckpt_path)
            shutil.rmtree(ckpt_path)
        