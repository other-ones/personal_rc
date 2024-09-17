# ------------------------------------------
# TextDiffuser: Diffusion Models as Text Painters
# Paper Link: https://arxiv.org/abs/2305.10855
# Code Link: https://github.com/microsoft/unilm/tree/master/textdiffuser
# Copyright (c) Microsoft Corporation.
# This file provides the inference script.
# ------------------------------------------

import json
import os
import numpy as np
import argparse
from clipscore import cal_clipscore
# from fid_score import calculate_fid_given_paths


def eval_clipscore(pred_root, caption_path, device="cuda:0"):
    image_list=[]
    image_ids=[]
    text_list=[]
    json_data=json.load(open(caption_path))
    for file in json_data:
        caption=json_data[file]
        caption=caption.strip()
        image_list.append(os.path.join(pred_root,file+'.jpg'))
        image_ids.append(file)
        text_list.append(caption)

    clip_scores = []
    scores = []
    score = cal_clipscore(image_ids=image_ids, image_paths=image_list, text_list=text_list, device=device)
    clip_score = np.mean([s['CLIPScore'] for s in score.values()])
    clip_scores.append(clip_score)
    scores.append(score)
    model_name=pred_root.split('/')[-2]
    print("{}\t{}".format(model_name,np.mean(clip_scores)))
    return np.mean(clip_scores), scores


def evaluate_results(pred_root,caption_path):
    dataset_res = {}
    dataset_res['clipscore'], dataset_res['scores'] =\
            eval_clipscore(pred_root, caption_path, device="cuda:0")

    # method_res[method] = dataset_res
    # with open(os.path.join(pred_root, 'eval.json'), 'w') as fw:
    #     json.dump(dataset_res, fw)





if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    exps=open('explist').readlines()
    exps=[exp.strip().split('/')[-1] for exp in exps if exp.strip()]
    concepts=os.listdir('/data/twkim/diffusion/personalization/collected/images/')
    for exp in exps:
        # captions.json
        if exp.startswith('#'):
            continue
        cur_concept=None
        for concept in concepts:
            if concept in exp and ('pet_'+concept) not in exp:
                cur_concept=concept
        pred_root=os.path.join('../../results/single/',cur_concept,exp,'generated')
        num_samples=len(os.listdir(pred_root))
        caption_path=os.path.join('../../results/single/',cur_concept,exp,'captions.json')
        # print(caption_path,'caption_path')
        caption_path=caption_path.replace('_regen7777','')
        if not (os.path.exists(caption_path) and num_samples>10):
            print('cont')
            continue

        evaluate_results(pred_root, caption_path)
