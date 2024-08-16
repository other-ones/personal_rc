import os
root='results/single'
concepts=os.listdir(root)
for concept in concepts:
    concept_path=os.path.join(root,concept)
    exps=os.listdir(concept_path)
    for exp in exps:
        exp_path=os.path.join(concept_path,exp)
        for type in ['dino','masked_dino']:
            score_path=os.path.join(exp_path,'{}.json'.format(type))
            if os.path.exists(score_path):
                print(score_path)
                os.remove(score_path)
