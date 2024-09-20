import os
import numpy as np
src_root='/home/twkim/project/rich_context/custom_diffusion/results/cd_results/init_seed2940_qlab03_rep2'
dst_root='/home/twkim/project/rich_context/custom_diffusion/results/cd_results/init_seed2940_qlab03_rep2_valid'
os.makedirs(dst_root,exist_ok=True)
concepts=os.listdir(src_root)
for concept in concepts:
    src_concept_path=os.path.join(src_root,concept)
    dst_concept_path=os.path.join(dst_root,concept)
    os.makedirs(dst_concept_path,exist_ok=True)

    exps=os.listdir(src_concept_path)
    for exp in exps:
        src_exp_path=os.path.join(src_concept_path,exp)
        dst_exp_path=os.path.join(dst_concept_path,exp)
        if not '_ti' in src_exp_path:
            if not os.path.exists(dst_exp_path):
                os.symlink(src_exp_path,dst_exp_path)
            continue

        command_path=os.path.join(src_exp_path,'src/command.txt')
        lines=open(command_path).readlines()
        # cd_init_qlab03_mlm00005_backpack_dog_mprob015_mbatch25_mtarget_masked_lr1e4_tbatch1_ti500_s2000
        exp_splits=exp.split('_')
        ti_step_split=exp_splits[-2]
        # print(ti_step_split,'ti_step_split')
        assert 'ti' in ti_step_split
        denoted_resume_step=int(ti_step_split.replace('ti',''))
        for line in lines:
            line=line.strip()
            # --resume_cd_path=saved_models/cd_models/init_seed2940_qlab03_rep2/backpack_dog/cd_init_qlab03_mlm00001_backpack_dog_mprob015_mbatch25_mtarget_masked_lr1e5/checkpoints/checkpoint-250/custom_diffusion.pt
            if 'resume_cd_path' in line:
                splits=line.split('/')
                target_split=splits[-2]
                actual_resume_step=int(target_split.split('-')[-1])
                if actual_resume_step != denoted_resume_step:
                    # print(exp)
                    continue
                else:
                    print(exp)
                    if not os.path.exists(dst_exp_path):
                        os.symlink(src_exp_path,dst_exp_path)