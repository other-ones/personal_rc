import os
import shutil
src='/home/twkim/project/rich_context/textual_inversion/results/ti_results/single_reduced4_noprior_seed7777_rep1_qlab03'
dst='/home/twkim/project/rich_context/textual_inversion/results/ti_results/single_reduced4_noprior_seed7777_rep2_qlab03'


concepts=os.listdir(src)
for concept in concepts:
    src_concept_path=os.path.join(src,concept)
    dst_concept_path=os.path.join(dst,concept)
    if not os.path.exists(dst_concept_path):
        continue
    src_exp_path=os.path.join(src_concept_path,'ti_qlab03_noprior_nomlm_{}_s3000'.format(concept))
    dst_exp_path=os.path.join(dst,concept,'ti_qlab03_noprior_nomlm_{}_s3000'.format(concept))
    if not os.path.exists(dst_exp_path):
        print(concept,'linked')
        os.symlink(src_exp_path,dst_exp_path)
