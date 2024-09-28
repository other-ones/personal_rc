import os
import shutil
mlm_list=['nomlm','_mlm00001_','_mlm0001_']
src_root='/home/twkim/project/rich_context/textual_inversion/saved_models/ti_models/bigger_reduced4_prior_seed7777_qlab03_rep2'
dst_root='/home/twkim/project/rich_context/textual_inversion/saved_models/ti_attn_models/bigger_reduced4_prior_seed7777_qlab03_rep2'
os.makedirs(dst_root,exist_ok=True)
concepts=os.listdir(src_root)
for concept in concepts:
    src_concept_path=os.path.join(src_root,concept)
    dst_concept_path=os.path.join(dst_root,concept)
    exps=os.listdir(src_concept_path)
    os.makedirs(dst_concept_path,exist_ok=True)
    for exp in exps:
        src_exp_path=os.path.join(src_concept_path,exp)
        dst_exp_path=os.path.join(dst_concept_path,exp)
        valid=False
        for mlm in mlm_list:
            if mlm in exp:
                valid=True
                break
        if not valid:
            continue
        os.makedirs(dst_exp_path,exist_ok=True)
        src_ckpt_root=os.path.join(src_exp_path,'checkpoints')
        dst_ckpt_root=os.path.join(dst_exp_path,'checkpoints')
        if not os.path.exists(dst_ckpt_root):
            shutil.copytree(src_ckpt_root,dst_ckpt_root)
            print(dst_ckpt_root)