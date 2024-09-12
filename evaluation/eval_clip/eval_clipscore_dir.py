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

def extract_values(exp):
    # Determine if "nomlm" is present
    is_nomlm = 'nomlm' in exp

    # Determine if "tagged" is present
    tagged = 'tagged' in exp

    # Extract mlm, lr, and step values using regex
    mlm_match = re.search(r'_mlm(\d+)_', exp)
    lr_match = re.search(r'_lr(\d+e\d+)_', exp)
    s_match = re.search(r'_s(\d+)$', exp)

    # Default values if not found
    mlm = (mlm_match.group(1))[::-1] if mlm_match else 'inf'
    lr = float(lr_match.group(1).replace('e', 'e-')) if lr_match else float('inf')
    step = int(s_match.group(1)) if s_match else float('inf')

    # Return a tuple for sorting with priority: is_nomlm, mlm, lr, step, no_tagged
    return (not is_nomlm, mlm, lr, step, tagged)
def eval_clipscore(pred_root, caption_path, device="cuda:0",num_samples=None):
    image_list=[]
    image_ids=[]
    text_list=[]
    json_data=json.load(open(caption_path))
    keys=list(json_data.keys())
    for key in keys:
        file=key.split('.')[0]
        if num_samples is not None and int(file)>num_samples:
            continue
        caption=json_data[file]
        caption=caption.strip()
        image_list.append(os.path.join(pred_root,file+'.jpg'))
        image_ids.append(file)
        text_list.append(caption)
    # print(len(image_list),'len(image_list)')
    # print(len(text_list),'len(text_list)')
    clip_scores = []
    scores = []
    score = cal_clipscore(image_ids=image_ids, image_paths=image_list, text_list=text_list, device=device)
    clip_score = np.mean([s['CLIPScore'] for s in score.values()])
    clip_scores.append(clip_score)
    scores.append(score)
    model_name=pred_root.split('/')[-2]
    print("{}\t{}".format(model_name,np.mean(clip_scores)))
    return np.mean(clip_scores), scores


def evaluate_results(pred_root,caption_path,num_samples):
    dataset_res = {}
    dataset_res['clipscore'], dataset_res['scores'] =\
            eval_clipscore(pred_root, caption_path, device="cuda:0",num_samples=num_samples)

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
    parser.add_argument('--keywords',type=str)
    parser.add_argument('--num_samples',type=int)
    parser.add_argument('--exclude',type=str)
    parser.add_argument('--ignore_legacy',type=int)
    args=parser.parse_args()
    if args.ignore_legacy:
        inp=input('IGNORE EXISTING RESULTS? y/n')
        if inp!='y':
            print('Unintended argument passed')
            exit()



    # /home/twkim/project/textual_inversion/results/single_normalized/tmp
    dir_path=args.dir_path
    keywords=args.keywords
    num_samples=args.num_samples
    concepts=os.listdir(dir_path)
    if keywords is not None:
        keywords=keywords.split('-')
    
    for concept in concepts:
        # if 'chair' in concept:
        #     continue
        print(concept)
        concept_path=os.path.join(dir_path,concept)
        exps=os.listdir(concept_path)
        sorted_exps = sorted(exps, key=extract_values)
        if any("_ti" in exp or "nomlm" in exp for exp in sorted_exps):
            for exp in sorted_exps:
                if args.exclude is not None and args.exclude in exp:
                        continue
                exp_path=os.path.join(concept_path,exp)
                pred_root=os.path.join(exp_path,'generated')
                result_path=os.path.join(exp_path, 'clip.json')
                model_name=pred_root.split('/')[-2]
                if os.path.exists(result_path) and (args.ignore_legacy==0):
                    dataset_res=json.load(open(result_path))
                    if num_samples:
                        score_list=dataset_res['scores'][0]
                        scores=[]
                        for fname in score_list:
                            if int(fname)>num_samples:
                                continue
                            print(score_list[fname],'score_list[fname]')
                            scores.append(score_list[fname])
                        print('{}\t{}\tnum_samples:{}'.format(model_name,np.mean(scores),num_samples))
                    else:
                        print('{}\t{}'.format(model_name,dataset_res['clipscore']))
                    continue
                caption_path=os.path.join(exp_path,'captions.json')
                if not os.path.exists(caption_path):
                    continue
                fsize=os.stat(caption_path).st_size
                if fsize==0:
                    continue
                dataset_res=evaluate_results(pred_root, caption_path,num_samples=None)
                with open(result_path, 'w') as fw:
                    json.dump(dataset_res, fw)
            print()
