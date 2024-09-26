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
meta_info={
    # train_prior/eval_prior/train_prompt_type/eval_prompt_type
    'duck_toy':('duck','duck toy','nonliving','nonliving'),
    'chair1': ('chair','chair','nonliving','nonliving'),
    'dog6': ('dog','dog','pet','living'),
    'teapot':('teapot','teapot','nonliving','nonliving'),
    'cat1': ('cat','cat','pet','living'),

    'pet_cat1':('cat','cat','pet','living'),
    'wooden_pot':('pot','wooden pot','nonliving','nonliving'),
    'backpack_dog':('backpack','backpack','nonliving','nonliving'),
    'poop_emoji':('toy','toy','nonliving','nonliving'),
    'cat2':('cat','cat','pet','living'),
    'dog3':  ('dog','dog','pet','living'),
    'pet_dog1':('dog','dog','pet','living'),
    'backpack':('backpack','backpack','nonliving','nonliving'),
    'teddybear':('teddy','teddy bear','nonliving','nonliving'),
    'cat_statue': ('toy','toy','nonliving','nonliving'),
    'rc_car':('toy','toy','nonliving','nonliving'),
}
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
            command_path=os.path.join(exp_path,'src/command.txt')
            lines=open(command_path).readlines()
            for line in lines:
                line=line.strip()
                if '--train_prior_concept1' in line:
                    if not 'teddy' in line:
                        print(exp_path)

