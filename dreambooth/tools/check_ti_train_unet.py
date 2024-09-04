import os
import numpy as np
root='../saved_models/db_models/single_capv7_seed2940_rep1/'
concepts=os.listdir(root)
for concept in concepts:
    concept_path=os.path.join(root,concept)
    exps=os.listdir(concept_path)
    for exp in exps:
        if not '_ti' in exp:
            continue
        exp_path=os.path.join(concept_path,exp)
        unet_exp_name=exp.replace('lr5e4','lr1e6')
        unet_exp_name=unet_exp_name.replace('lr1e4','lr1e6')
        unet_exp_name=unet_exp_name.replace('_ti','')
        resume_unet_path=os.path.join('saved_models/db_models/single_capv7_seed2940_rep1',concept,unet_exp_name,'checkpoints/checkpoint-1000/unet_s1000.pt')
        command_path=os.path.join(exp_path,'src/command.txt')
        lines=open(command_path)
        found=False
        unet_line=None
        for line in lines:
            line=line.strip()
            if 'resume_unet_path' in line:
                unet_line=line
                if resume_unet_path in line:
                    found=True
        if not found:
            print(exp,'exp')
            print(unet_path,'unet_path')
            print(unet_line,'unet_line')
        assert found,unet_path




print('NO ERROR')