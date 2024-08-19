import shutil
import os
root='results/single_seed2940'
concepts=os.listdir(root)
for concept in concepts:
    concept_path=os.path.join(root,concept)
    exps=os.listdir(concept_path)
    for exp in exps:
        exp_path=os.path.join(concept_path,exp)
        masked_root=os.path.join(exp_path,'masked')
        mask_root=os.path.join(exp_path,'mask')
        log_path=os.path.join(exp_path,'result.txt')
        if os.path.exists(masked_root):
            print(masked_root)
            shutil.rmtree(masked_root)
        if os.path.exists(mask_root):
            print(mask_root)
            shutil.rmtree(mask_root)
        if os.path.exists(log_path):
            print(log_path)
            os.remove(log_path)

        # for type in ['dino','masked_dino']:
        #     score_path=os.path.join(exp_path,'{}.json'.format(type))
        #     if os.path.exists(score_path):
        #         print(score_path)
        #         os.remove(score_path)
