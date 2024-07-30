import shutil
root='saved_models/disenbooth_models/sd2/single'
concepts=os.listdir(root)
for concept in concepts:
    concept_path=os.path.join(root,concept)
    exps=os.listdir(concet_path)
    for exp in exps:
        exp_path=os.path.join(concept_path,exp)
