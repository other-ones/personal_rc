import os
import shutil
root='saved_models/pplus_models/single'
keyword='mlm'
concepts=os.listdir(root)
for concept in concepts:
    concept_path=os.path.join(root,concept)
    exps=os.listdir(concept_path)
    for exp in exps:
        if not keyword in exp:
            continue
        exp_path=os.path.join(concept_path,exp)
        print(exp_path)
        shutil.rmtree(exp_path)