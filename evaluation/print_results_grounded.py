
import json
import os
import numpy as np
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--dir_path')
parser.add_argument('--exclude')
parser.add_argument('--include')
args=parser.parse_args()
dir_path=args.dir_path
exclude=args.exclude
include=args.include

concepts=os.listdir(dir_path)
clip_score_dict={}
masked_dino_score_dict={}
for concept in concepts:
    concept_path=os.path.join(dir_path,concept)
    exps=os.listdir(concept_path)
    exps=sorted(exps)[::-1]
    
    print('{}\t{}\t{}\t{}\t{}'.format(concept,'CLIP','DeltaCLIP','GDINO','DeltaGDINO'))
    nomlm_clip_score=json.load(open(os.path.join(concept_path,'custom_nomlm_{}_s250'.format(concept),'clip.json')))['clipscore']
    nomlm_masked_dino_score=json.load(open(os.path.join(concept_path,'custom_nomlm_{}_s250'.format(concept),'masked_dino.json')))['masked_dino']
    # print('{}\t{}\t{}\t{}\t{}'.format('custom_nomlm_{}_s250'.format(concept),nomlm_clip_score,'-',nomlm_masked_dino_score,'-'))

    exp_key='custom_nomlm_s250'.format(concept)
    if exp_key in masked_dino_score_dict:
        clip_score_dict[exp_key].append(nomlm_clip_score)
        masked_dino_score_dict[exp_key].append(nomlm_masked_dino_score)
    else:
        clip_score_dict[exp_key]=[nomlm_clip_score]
        masked_dino_score_dict[exp_key]=[nomlm_masked_dino_score]
    for exp in exps:
        # if 'nomlm' in exp:
        #     continue
        exp_key=exp.replace('{}_'.format(concept),'')
        exp_path=os.path.join(concept_path,exp)
        masked_dino_path=os.path.join(exp_path,'masked_dino.json')
        clip_path=os.path.join(exp_path,'clip.json')
        clip_score=json.load(open(clip_path))['clipscore']
        delta_clip=clip_score-nomlm_clip_score
        masked_dino_score=json.load(open(masked_dino_path))['masked_dino']
        delta_masked_dino=masked_dino_score-nomlm_masked_dino_score
        print('{}\t{}\t{}\t{}\t{}'.format(exp,clip_score,delta_clip,masked_dino_score,delta_masked_dino))
        if exp_key in clip_score_dict:
            clip_score_dict[exp_key].append(clip_score)
            masked_dino_score_dict[exp_key].append(masked_dino_score)
        else:
            clip_score_dict[exp_key]=[clip_score]
            masked_dino_score_dict[exp_key]=[masked_dino_score]
    print()



print('{}\t{}\t{}\t{}\t{}'.format('avg','CLIP','DeltaCLIP','GDINO','DeltaGDINO'))
avg_nomlm_clip_score=np.mean(clip_score_dict['custom_nomlm_s250'])
avg_nomlm_masked_dino_score=np.mean(masked_dino_score_dict['custom_nomlm_s250'])
# print('{}\t{}\t{}\t{}\t{}'.format('custom_nomlm_s250',avg_nomlm_clip_score,'-',avg_nomlm_masked_dino_score,'-'))
for key in clip_score_dict:
    # if 'nomlm' in key:
    #     continue
    avg_clip_score=np.mean(clip_score_dict[key])
    avg_masked_dino_score=np.mean(masked_dino_score_dict[key])
    delta_clip=avg_clip_score-avg_nomlm_clip_score
    delta_masked_dino=avg_masked_dino_score-avg_nomlm_masked_dino_score
    print('{}\t{}\t{}\t{}\t{}'.format(key,avg_clip_score,delta_clip,avg_masked_dino_score,delta_masked_dino))

    
    










assert len(clip_score_dict['custom_nomlm_s250'])==len(masked_dino_score_dict['custom_nomlm_s250'])
num_nomlm_score=len(clip_score_dict['custom_nomlm_s250'])
for key in clip_score_dict:
    len(clip_score_dict[key])==num_nomlm_score

for key in masked_dino_score_dict:
    len(clip_score_dict[key])==num_nomlm_score

