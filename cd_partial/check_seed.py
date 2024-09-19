import os
seed=7777
# 1. check saved_models
root1='saved_models/custom_diffusion/single_seed{}'.format(seed)
concepts=os.listdir(root1)
for concept in concepts:
    concept_path=os.path.join(root1,concept)
    exps=os.listdir(concept_path)
    for exp in exps:
        exp_path=os.path.join(concept_path,exp)
        command_path=os.path.join(exp_path,'src/command.txt')
        found=False
        lines=open(command_path).readlines()
        for line in lines:
            if '--seed={}'.format(seed) in line:
                found=True
                break
        assert found,exp_path


# 2. check results
root2='results/single_seed{}'.format(seed)
concepts=os.listdir(root2)
for concept in concepts:
    concept_path=os.path.join(root2,concept)
    exps=os.listdir(concept_path)
    for exp in exps:
        exp_path=os.path.join(concept_path,exp)
        command_path=os.path.join(exp_path,'src/command.txt')
        found=False
        lines=open(command_path).readlines()
        for line in lines:
            if '--seed={}'.format(seed) in line:
                found=True
                break
        assert found,exp_path

print('no error')