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


def eval_clipscore(pred_root, caption_path,target_subset, device="cuda:0"):
    image_list=[]
    image_ids=[]
    text_list=[]
    json_data=json.load(open(caption_path))
    for file in json_data:
        caption=json_data[file]
        caption=caption.strip()
        if target_subset is not None and target_subset not in file:
            continue
        image_list.append(os.path.join(pred_root,file+'.jpg'))
        image_ids.append(file)
        text_list.append(caption)

    clip_scores = []
    scores = []
    score = cal_clipscore(image_ids=image_ids, image_paths=image_list, text_list=text_list, device=device)
    # pred_list_file=open(os.path.join(pred_root,'clip.txt'),'w')
    # for item in score.values():
    #     pred_list_file.write('{}\n'.format(item['CLIPScore']))

    clip_score = np.mean([s['CLIPScore'] for s in score.values()])
    clip_scores.append(clip_score)
    scores.append(score)
    model_name=pred_root.split('/')[-2]
    print("{}\t{}".format(model_name,np.mean(clip_scores)), clip_scores)
    return np.mean(clip_scores), scores


def evaluate_results(pred_root,caption_path,target_subset):
    dataset_res = {}
    dataset_res['clipscore'], dataset_res['scores'] =\
            eval_clipscore(pred_root, caption_path,target_subset, device="cuda:0")

    # method_res[method] = dataset_res
    # with open(os.path.join(pred_root, 'eval.json'), 'w') as fw:
    #     json.dump(dataset_res, fw)





if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--pred_root')
    parser.add_argument('--caption_path')
    parser.add_argument('--target_subset')
    args=parser.parse_args()
    evaluate_results(args.pred_root, args.caption_path,args.target_subset)
