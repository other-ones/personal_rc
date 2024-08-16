import shutil
import os
root='results/single'
concepts=os.listdir(root)
for concept in concepts:
    concept_path=os.path.join(root,concept)
    exps=os.listdir(concept_path)
    for exp in exps:
        exp_path=os.path.join(concept_path,exp)
        img_root=os.path.join(exp_path,'generated')
        log_path=os.path.join(exp_path,'result.txt')
        images=os.listdir(img_root)
        if (not len(images)) and not(os.path.exists(log_path)):
            print(exp_path)
            shutil.rmtree(exp_path)
        # for type in ['dino','masked_dino']:
        #     score_path=os.path.join(exp_path,'{}.json'.format(type))
        #     if os.path.exists(score_path):
        #         print(score_path)
        #         os.remove(score_path)
