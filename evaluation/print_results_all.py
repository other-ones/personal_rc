import re

import json
import os
import numpy as np
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--dir_path')
parser.add_argument('--include')
parser.add_argument('--exclude')
parser.add_argument('--keywords')
parser.add_argument('--grounded',default=0,type=int)
parser.add_argument('--strict',default=0,type=int)
args=parser.parse_args()
dir_path=args.dir_path
include=args.include
exclude=args.exclude
keywords=args.keywords
grounded=args.grounded

concepts=os.listdir(dir_path)
clip_score_dict={}
dino_score_dict={}
fg_dino_score_dict={}
bg_dino_score_dict={}
if keywords is not None:
    keywords=keywords.split('-')
if exclude is not None:
    exclude=exclude.split('-')

def extract_values(exp):
    # Determine if "nomlm" is present
    is_nomlm = 'nomlm' in exp
    # Extract mlm, lr, and step values using regex
    mlm_match = re.search(r'_mlm(\d+)_', exp)
    lr_match = re.search(r'_lr(\d+e\d+)_', exp)
    s_match = re.search(r'_s(\d+)$', exp)
    mprob_match = re.search(r'_mprob(\d+)_', exp)

    # Default values if not found
    mlm = (mlm_match.group(1))[::-1] if mlm_match else 'inf'
    lr = float(lr_match.group(1).replace('e', 'e-')) if lr_match else float('inf')
    step = -int(s_match.group(1)) if s_match else float('inf')
    mprob = (mprob_match.group(1))[::-1] if mprob_match else 'inf'

    # Return a tuple for sorting with priority: is_nomlm, mlm, lr, step, no_tagged
    return (not step,is_nomlm,mlm,mprob,lr)


    
for concept in concepts:
    concept_path=os.path.join(dir_path,concept)
    exps=os.listdir(concept_path)
    exps=sorted(exps,key=extract_values)
    for exp in exps:
        # if keywords is not None:
        #     valid1=True
        #     for keyword in keywords:
        #         if keyword not in exp_path:
        #             valid1=False
        #             break
        # else:
        #     valid1=True
        # if exclude is not None:
        #     valid2=True
        #     for item in exclude:
        #         if item in exp_path:
        #             valid2=False
        #             break
        # else:
        #     valid2=True

        exp_key=exp.replace('{}_'.format(concept),'')
        exp_path=os.path.join(concept_path,exp)
        dino_path=os.path.join(exp_path,'dino.json')
        fg_dino_path=os.path.join(exp_path,'masked_dino.json')
        bg_dino_path=os.path.join(exp_path,'masked_dino_bg.json')
        clip_path=os.path.join(exp_path,'clip.json')
        if not(os.path.exists(dino_path) and os.path.exists(clip_path) and os.path.exists(fg_dino_path) and os.path.exists(bg_dino_path)):
            print(exp_path,'exp')
            continue
        clip_score=json.load(open(clip_path))['clipscore']
        dino_score=json.load(open(dino_path))['dino']
        fg_dino_score=json.load(open(fg_dino_path))['masked_dino']
        bg_dino_score=json.load(open(bg_dino_path))['masked_dino_bg']
        # print(f'{exp}\t{clip_score}\t{dino_score}\t{fg_dino_score}\t{bg_dino_score}')
        if exp_key in clip_score_dict:
            clip_score_dict[exp_key].append(clip_score)
            dino_score_dict[exp_key].append(dino_score)
            fg_dino_score_dict[exp_key].append(fg_dino_score)
            bg_dino_score_dict[exp_key].append(bg_dino_score)
        else:
            clip_score_dict[exp_key]=[clip_score]
            dino_score_dict[exp_key]=[dino_score]
            fg_dino_score_dict[exp_key]=[fg_dino_score]
            bg_dino_score_dict[exp_key]=[bg_dino_score]
    # print()



print(args.dir_path.split('/')[-1])
print('AVG\tCLIP\tDINO\tDINO-FG\tDINO-BG')
for key in clip_score_dict:
    avg_clip_score=np.mean(clip_score_dict[key])
    avg_dino_score=np.mean(dino_score_dict[key])
    avg_fg_dino_score=np.mean(fg_dino_score_dict[key])
    avg_bg_dino_score=np.mean(bg_dino_score_dict[key])
    l1=len(clip_score_dict[key])
    l2=len(dino_score_dict[key])
    l3=len(fg_dino_score_dict[key])
    l4=len(bg_dino_score_dict[key])
    # if not (l1 in[15,16] and l2 in [15,16] and l3 in [15,16] and l4 in [15,16]):
    if not (l1 in[16] and l2 in [16] and l3 in [16] and l4 in [16]):
        print(key,l1,l2,l3,l4)
        continue
    assert l1==l2
    assert l1==l3
    assert l1==l4
    print(f'{key}\t{avg_clip_score}\t{avg_dino_score}\t{avg_fg_dino_score}\t{avg_bg_dino_score}')

    
    










