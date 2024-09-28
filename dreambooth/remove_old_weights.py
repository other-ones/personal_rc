import shutil
import os
root='saved_models/db_models/old'
dirs=os.listdir(root)
for dir in dirs:
    dir_path=os.path.join(root,dir)
    concepts=os.listdir(dir_path)
    for concept in concepts:
        concept_path=os.path.join(dir_path,concept)
        exps=os.listdir(concept_path)
        for exp in exps:
            exp_path=os.path.join(concept_path,exp)
            ckpt_root=os.path.join(exp_path,'checkpoints')
            shutil.rmtree(ckpt_root)
            print(exp_path,'removed')