import re
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
    return dataset_res


def extract_mlm_step(name):
    mlm_match = re.search(r'mlm(\d+)', name)
    step_match = re.search(r's(\d+)', name)
    if mlm_match and step_match:
        mlm_number = int(mlm_match.group(1))
        step_number = int(step_match.group(1))
        return (mlm_number, step_number)
    return (None, None)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--dir_path',type=str)
    parser.add_argument('--target_keyword',type=str)

    # /home/twkim/project/textual_inversion/results/single_normalized/tmp
    args=parser.parse_args()
    dir_path=args.dir_path
    target_keyword=args.target_keyword
    concepts=os.listdir(dir_path)
    for concept in concepts:
        print(concept)
        concept_path=os.path.join(dir_path,concept)
        exps=os.listdir(concept_path)
        # exps=sorted(exps)[::-1]
        # exps=sorted(exps)[:]
        # if not '_mlm' in exps[0]:
        # exps = sorted(exps, key=extract_mlm_step)
        for exp in exps:
            if target_keyword is not None and (target_keyword not in exp):
                continue
            exp_path=os.path.join(concept_path,exp)
            pred_root=os.path.join(exp_path,'generated')
            result_path=os.path.join(exp_path, 'clip.json')
            model_name=pred_root.split('/')[-2]
            if os.path.exists(result_path):
                dataset_res=json.load(open(result_path))
                print('{}\t{}'.format(model_name,dataset_res['clipscore']))
                continue
            caption_path=os.path.join(exp_path,'captions.json')
            if not os.path.exists(caption_path):
                continue
            fsize=os.stat(caption_path).st_size
            if fsize==0:
                continue
            num_samples=len(os.listdir(pred_root))
            dataset_res=evaluate_results(pred_root, caption_path)
            with open(result_path, 'w') as fw:
                json.dump(dataset_res, fw)

        print()
