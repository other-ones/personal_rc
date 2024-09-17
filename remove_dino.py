import os
root1='/home/twkim/project/rich_context/pplus/results/pplus_results/bigger_reduced4_prior_seed7777_qlab03_rep1'
root2='/home/twkim/project/rich_context/pplus/results/ti_results/bigger_reduced4_prior_seed7777_qlab03_rep2'
for root in [root1,root2]:
    concepts=os.listdir(root)
    for concept in concepts:
        concept_path=os.path.join(root,concept)
        exps=os.listdir(concept_path)
        for exp in exps:
            exp_path=os.path.join(concept_path,exp)
            dino_path=os.path.join(exp_path,'masked_dino.json')
            if not os.path.exists(dino_path):
                continue
            os.remove(dino_path)
            print(dino_path,'deleted')