
import json
import os
import numpy as np
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--dir_path')
parser.add_argument('--include')
parser.add_argument('--exclude')
args=parser.parse_args()
dir_path=args.dir_path
include=args.include
exclude=args.exclude

concepts=os.listdir(dir_path)
clip_score_dict={}
dino_score_dict={}
for concept in concepts:
    concept_path=os.path.join(dir_path,concept)
    exps=os.listdir(concept_path)
    exps=sorted(exps)[::-1]
    
    print('{}\t{}\t{}\t{}\t{}'.format(concept,'CLIP','DeltaCLIP','DINO','DeltaDINO'))
    nomlm_clip_score=json.load(open(os.path.join(concept_path,'custom_nomlm_{}_s250'.format(concept),'clip.json')))['clipscore']
    nomlm_dino_score=json.load(open(os.path.join(concept_path,'custom_nomlm_{}_s250'.format(concept),'dino.json')))['dino']
    print('{}\t{}\t{}\t{}\t{}'.format('custom_nomlm_{}_s250'.format(concept),nomlm_clip_score,'-',nomlm_dino_score,'-'))
    exp_key='custom_nomlm_s250'.format(concept)
    if exp_key in dino_score_dict:
        clip_score_dict[exp_key].append(nomlm_clip_score)
        dino_score_dict[exp_key].append(nomlm_dino_score)
    else:
        clip_score_dict[exp_key]=[nomlm_clip_score]
        dino_score_dict[exp_key]=[nomlm_dino_score]
    for exp in exps:
        if exclude in exp:
            continue
        if 'nomlm_{}_s250'.format(concept) in exp:
            continue
        exp_key=exp.replace('{}_'.format(concept),'')
        exp_path=os.path.join(concept_path,exp)
        dino_path=os.path.join(exp_path,'dino.json')
        clip_path=os.path.join(exp_path,'clip.json')
        if not(os.path.exists(dino_path) and os.path.exists(clip_path)):
            # print(os.path.exists(clip_path),'os.path.exists(clip_path)')
            # print(os.path.exists(dino_path),'os.path.exists(dino_path)')
            continue
        clip_score=json.load(open(clip_path))['clipscore']
        delta_clip=clip_score-nomlm_clip_score
        dino_score=json.load(open(dino_path))['dino']
        delta_dino=dino_score-nomlm_dino_score
        print('{}\t{}\t{}\t{}\t{}'.format(exp,clip_score,delta_clip,dino_score,delta_dino))
        if exp_key in clip_score_dict:
            clip_score_dict[exp_key].append(clip_score)
            dino_score_dict[exp_key].append(dino_score)
        else:
            clip_score_dict[exp_key]=[clip_score]
            dino_score_dict[exp_key]=[dino_score]
    print()



print('{}\t{}\t{}\t{}\t{}'.format('avg','CLIP','DeltaCLIP','DINO','DeltaDINO'))
avg_nomlm_clip_score=np.mean(clip_score_dict['custom_nomlm_s250'])
avg_nomlm_dino_score=np.mean(dino_score_dict['custom_nomlm_s250'])
# print('{}\t{}\t{}\t{}\t{}'.format('custom_nomlm_s250',avg_nomlm_clip_score,'-',avg_nomlm_dino_score,'-'))
for key in clip_score_dict:
    # if 'nomlm' in key:
    #     continue
    avg_clip_score=np.mean(clip_score_dict[key])
    avg_dino_score=np.mean(dino_score_dict[key])
    delta_clip=avg_clip_score-avg_nomlm_clip_score
    delta_dino=avg_dino_score-avg_nomlm_dino_score
    print('{}\t{}\t{}\t{}\t{}'.format(key,avg_clip_score,delta_clip,avg_dino_score,delta_dino))

    
    










assert len(clip_score_dict['custom_nomlm_s250'])==len(dino_score_dict['custom_nomlm_s250'])
num_nomlm_score=len(clip_score_dict['custom_nomlm_s250'])
for key in clip_score_dict:
    # print(num_nomlm_score,clip_score_dict[key],key)
    assert len(clip_score_dict[key])==num_nomlm_score

for key in dino_score_dict:
    assert len(clip_score_dict[key])==num_nomlm_score

