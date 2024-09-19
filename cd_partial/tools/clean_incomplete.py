import shutil
import os
root='saved_models/custom_diffusion/single_re'
concepts=os.listdir(root)
for concept in concepts:
    concept_path=os.path.join(root,concept)
    exps=os.listdir(concept_path)
    for exp in exps:
        exp_path=os.path.join(concept_path,exp)
        bin_path=os.path.join(exp_path,'checkpoints/checkpoint-250/<{}>.bin'.format(concept))
        pt_path=os.path.join(exp_path,'checkpoints/checkpoint-250/custom_diffusion.pt')
        if not (os.path.exists(bin_path) and os.path.exists(pt_path)):
            print(exp_path)
            shutil.rmtree(exp_path)