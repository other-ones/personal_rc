
import json
import os
import numpy as np
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--dir_path')
parser.add_argument('--include')
parser.add_argument('--exclude')
parser.add_argument('--grounded',default=0,type=int)
args=parser.parse_args()
dir_path=args.dir_path
include=args.include
exclude=args.exclude
grounded=args.grounded

concepts=os.listdir(dir_path)
clip_score_dict={}
dino_score_dict={}
if args.grounded:
    score_name='masked_dino'
else:
    score_name='dino'
for concept in concepts:
    concept_path=os.path.join(dir_path,concept)
    exps=os.listdir(concept_path)
    exps=sorted(exps)[::-1]
    nomlm_clip_score=json.load(open(os.path.join(concept_path,'custom_nomlm_{}_s250'.format(concept),'clip.json')))['clipscore']
    if args.grounded:
        print('{}\t{}\t{}\t{}\t{}'.format(concept,'CLIP','Delta','GDINO','Delta'))
        nomlm_dino_score=json.load(open(os.path.join(concept_path,'custom_nomlm_{}_s250'.format(concept),'{}.json'.format(score_name))))[score_name]
    else:
        print('{}\t{}\t{}\t{}\t{}'.format(concept,'CLIP','Delta','DINO','Delta'))
        nomlm_dino_score=json.load(open(os.path.join(concept_path,'custom_nomlm_{}_s250'.format(concept),'{}.json'.format(score_name))))[score_name]
        print('{}\t{}\t{}\t{}\t{}'.format('custom_nomlm_{}_s250'.format(concept),nomlm_clip_score,'-',nomlm_dino_score,'-'))
    exp_key='custom_nomlm_s250'.format(concept)
    if exp_key in dino_score_dict:
        clip_score_dict[exp_key].append(nomlm_clip_score)
        dino_score_dict[exp_key].append(nomlm_dino_score)
    else:
        clip_score_dict[exp_key]=[nomlm_clip_score]
        dino_score_dict[exp_key]=[nomlm_dino_score]
    for exp in exps:
        if exclude is not None and exclude in exp:
            continue
        if 'nomlm_{}_s250'.format(concept) in exp:
            continue
        exp_key=exp.replace('{}_'.format(concept),'')
        exp_path=os.path.join(concept_path,exp)
        dino_path=os.path.join(exp_path,'{}.json'.format(score_name))
        clip_path=os.path.join(exp_path,'clip.json')
        if not(os.path.exists(dino_path) and os.path.exists(clip_path)):
            continue
        clip_score=json.load(open(clip_path))['clipscore']
        delta_clip=clip_score-nomlm_clip_score
        dino_score=json.load(open(dino_path))[score_name]
        delta_dino=dino_score-nomlm_dino_score
        print('{}\t{}\t{}\t{}\t{}'.format(exp,clip_score,delta_clip,dino_score,delta_dino))
        if exp_key in clip_score_dict:
            clip_score_dict[exp_key].append(clip_score)
            dino_score_dict[exp_key].append(dino_score)
        else:
            clip_score_dict[exp_key]=[clip_score]
            dino_score_dict[exp_key]=[dino_score]
    print()



print(args.dir_path.split('/')[-1])
if args.grounded:
    print('{}\t{}\t{}\t{}\t{}'.format('avg','CLIP','Delta','GDINO','Delta'))
else:
    print('{}\t{}\t{}\t{}\t{}'.format('avg','CLIP','Delta','DINO','Delta'))
avg_nomlm_clip_score=np.mean(clip_score_dict['custom_nomlm_s250'])
avg_nomlm_dino_score=np.mean(dino_score_dict['custom_nomlm_s250'])
num_nomlm_score=len(clip_score_dict['custom_nomlm_s250'])
for key in clip_score_dict:
    # if 'nomlm' in key:
    #     continue
    if len(clip_score_dict[key]) !=num_nomlm_score:
        print(key)
        continue
    if 'resume' in key:
        step=int(key.split('_s')[-1])
        if step>150:
            continue
    
    avg_clip_score=np.mean(clip_score_dict[key])
    avg_dino_score=np.mean(dino_score_dict[key])
    delta_clip=avg_clip_score-avg_nomlm_clip_score
    delta_dino=avg_dino_score-avg_nomlm_dino_score
    print('{}\t{}\t{}\t{}\t{}'.format(key,avg_clip_score,delta_clip,avg_dino_score,delta_dino))

    
    










# assert len(clip_score_dict['custom_nomlm_s250'])==len(dino_score_dict['custom_nomlm_s250'])
# for key in clip_score_dict:
#     # print(num_nomlm_score,clip_score_dict[key],key)
#     assert len(clip_score_dict[key])==num_nomlm_score

# for key in dino_score_dict:
#     assert len(clip_score_dict[key])==num_nomlm_score

