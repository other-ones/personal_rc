import os
import json
# # ti
# results/ti_results/bigger_reduced4_prior_seed7777_qlab03_rep2

# # pplus
# results/pplus_results/bigger_reduced4_prior_seed7777_qlab03_rep1

# # db
# results/db_results/bigger_seed7777_qlab03_rep1
# results/db_results/bigger_seed7777_qlab03_rep2

# # cd
# results/cd_results/highmlm_seed2940_qlab03_rep1
# results/cd_results/init_seed2940_qlab03_rep2

# # ablation
# results/ti_results/abl_mprob_prior_seed7777_qlab03_rep1
# results/ti_results/abl_prompt_size_prior_seed7777_qlab03_rep1
import os
import shutil

prefix='/data/twkim/diffusion/personalization/'
roots=[
    'results/ti_results/bigger_reduced4_prior_seed7777_qlab03_rep2',
    'results/pplus_results/bigger_reduced4_prior_seed7777_qlab03_rep1',
    'results/db_results/bigger_seed7777_qlab03_rep1',
    'results/db_results/bigger_seed7777_qlab03_rep2',
    'results/db_results/bigger_seed7777_qlab03_rep2',
    'results/cd_results/highmlm_seed2940_qlab03_rep1',
    'results/cd_results/init_seed2940_qlab03_rep2',
    'results/ti_results/abl_mprob_prior_seed7777_qlab03_rep1',
    'results/ti_results/abl_prompt_size_prior_seed7777_qlab03_rep1',
    ]
correct_caption_path='teddy_captions.json'
for root in roots:
    print('\n')
    print(root)
    src=os.path.join(prefix,root)
    concepts=os.listdir(src)
    for concept in concepts:
        if not 'teddybear' in concept:
            continue
        concept_path=os.path.join(src,concept)
        if not os.path.exists(concept_path):
            print(concept_path,'concept_path')
            continue
        exps=os.listdir(concept_path)
        for exp in exps:
            exp_path=os.path.join(concept_path,exp)
            caption_path=os.path.join(exp_path,'captions.json')
            clip_path=os.path.join(exp_path,'clip.json')
            if os.path.exists(caption_path):
                fsize=os.stat(caption_path).st_size
                if fsize==0:
                    print(caption_path,'NOT FINISHED')
                    continue
                data=json.load(open(caption_path))
                for key in data:
                    caption=data[key]
                    if 'teddy bear' not in caption:
                        print(caption_path)
                        shutil.copyfile(correct_caption_path,caption_path)
                        if os.path.exists(clip_path):
                            os.remove(clip_path)
                    break
            else:
                print(caption_path,'not exists')

