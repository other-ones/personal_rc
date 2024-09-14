import re
import json
import os
from pathlib import Path
import torchmetrics
from collections import defaultdict
from accelerate import Accelerator
from transformers import ViTImageProcessor, ViTModel
import torch
from torch.nn.functional import cosine_similarity
from PIL import Image

exclude_words_per_type={
    'pet':['wearing','outfit','a red','a purple','a shiny','a wet','a cube shaped'],
    'nonliving':['a red','a purple','a shiny','a wet','a cube shaped'],
    'building':['painting','artist','style','futuristic','surreal','close up shot','top view'],
    'flower':['painting','digital','style','color','dried'],
    'sunglasses':['style','artist','painting','digital','art piece',],
}
info_map={
    'backpack':('backpack','nonliving'),
    'teddybear':('bear','nonliving'),
    'wooden_pot':('pot','nonliving'),
    'vase':('vase','nonliving'),
    'cat1': ('cat','pet'),
    'pet_cat1':('cat','pet'),
    'pet_dog1':('dog','pet'),
    'barn': ('barn','building'),
    'chair1': ('chair','nonliving'),
    'cat_statue': ('toy','nonliving'),
    'rc_car':('toy','nonliving'),
    'pink_sunglasses':('sunglasses','sunglasses'),
    'dog3': ('dog','pet'),
    'dog6': ('dog','pet'),
    'flower1':('flower','flower'),

}


class DINOEvaluator:
    def __init__(self, device, dino_model='facebook/dino-vits16') -> None:
        self.device = device
        self.processor = ViTImageProcessor.from_pretrained(dino_model)
        self.model = ViTModel.from_pretrained(dino_model).to(device)

    @torch.inference_mode()
    def get_image_features(self, images) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(device=self.device)
        features = self.model(**inputs).last_hidden_state[:, 0, :]
        return features

    @torch.inference_mode()
    def img_to_img_similarity(self, src_images, generated_images):
        src_features = self.get_image_features(src_images)
        gen_features = self.get_image_features(generated_images)

        return torchmetrics.functional.pairwise_cosine_similarity(src_features, gen_features).mean().item()
def extract_mlm_step(name):
    print(name,'name')
    mlm_match = re.search(r'mlm(\d+)', name)
    step_match = re.search(r's(\d+)', name)
    if mlm_match and step_match:
        mlm_number = int(mlm_match.group(1))
        step_number = int(step_match.group(1))
        return (mlm_number, step_number)
    return (None, None)
def sort_by_step(items):
    def extract_step(item):
        # Split the string by '/' and take the last part before the score
        parts = item.split('/')
        # Find the part that contains 's' and extract the number
        for part in parts:
            if 's' in part:
                s_number = part.split('s')[-1]
                return int(s_number)
        return float('inf')  # Fallback, just in case (shouldn't happen with correct data)

    # Sort the list using the extracted s<number> as the key
    sorted_items = sorted(items, key=extract_step)
    return sorted_items
def sort_by_s_number(items):
    # Extract the number after 's' and use it as the key for sorting
    sorted_items = sorted(items, key=lambda x: int(x.split('_s')[1].split('\t')[0]))
    return sorted_items
def sort_by_mlm_and_s(items):
    # Extract mlm<number> and s<number> and use them as the key for sorting
    sorted_items = sorted(items, key=lambda x: (
        int(x.split('_mlm')[1].split('_')[0]),  # Extract mlm<number>
        int(x.split('_s')[1].split('\t')[0])    # Extract s<number>
    ))
    return sorted_items

def extract_values(exp):
    # Determine if "nomlm" is present
    is_nomlm = 'nomlm' in exp

    # Determine if "tagged" is present
    tagged = 'tagged' in exp

    # Extract mlm, lr, and step values using regex
    mlm_match = re.search(r'_mlm(\d+)_', exp)
    lr_match = re.search(r'_lr(\d+e\d+)_', exp)
    s_match = re.search(r'_s(\d+)$', exp)
    mprob_match = re.search(r'_mprob(\d+)_', exp)

    # Default values if not found
    mlm = (mlm_match.group(1))[::-1] if mlm_match else 'inf'
    lr = float(lr_match.group(1).replace('e', 'e-')) if lr_match else float('inf')
    step= int(s_match.group(1)) if s_match else float('inf')
    mprob = (mprob_match.group(1))[::-1] if mprob_match else 'inf'

    # Return a tuple for sorting with priority: is_nomlm, mlm, lr, step, no_tagged
    return (not is_nomlm,step,tagged, mprob, mlm, lr)
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--keywords',required=False)
    parser.add_argument('--dir_path',required=True)
    parser.add_argument('--grounded',type=int,default=0)
    parser.add_argument('--sort',type=int,default=0)
    parser.add_argument('--exclude_attr_change',type=int,default=0)
    parser.add_argument('--exclude',type=str)
    parser.add_argument('--ignore_legacy',type=int)
    args=parser.parse_args()
    if args.ignore_legacy:
        inp=input('IGNORE LEGACY RESULTS? (DINO) y/n')
        if inp!='y':
            print('Unintended argument passed')
            exit()
    dir_path=args.dir_path
    if args.keywords:
        keywords=args.keywords.split('-')
    else:
        keywords=None
    dino_eval=DINOEvaluator(device='cuda:0')
    results=[]
    concepts=os.listdir(dir_path)

    for concept in concepts:
        concept_path=os.path.join(dir_path,concept)
        exps=os.listdir(concept_path)
        if args.grounded:
            real_root=os.path.join('/data/twkim/diffusion/personalization/collected/masked/',concept)
        else:
            real_root=os.path.join('/data/twkim/diffusion/personalization/collected/images/',concept)
        # exps = sorted(exps, key=extract_mlm_step)
        concept_results=[]
        # exps=sorted(exps)[::-1]
        sorted_exps = sorted(exps, key=extract_values)
        # if any("_ti" in exp or "nomlm" in exp for exp in exps):
        if True:
            for exp in sorted_exps:
                if args.exclude is not None and args.exclude in exp:
                    continue
                exp_path=os.path.join(concept_path,exp)
                if keywords is not None:
                    valid=True
                    for keyword in keywords:
                        if keyword not in exp_path:
                            valid=False
                            break
                else:
                    valid=True
                if 'nomlm' in exp:
                    valid=True
                if not valid:
                    continue
                caption_path=os.path.join(exp_path,'captions.json')
                if not os.path.exists(caption_path):
                    continue
                fsize=os.stat(caption_path).st_size
                if fsize==0:
                    continue
                if args.exclude_attr_change:
                    cap_data=json.load(open(caption_path))
                    prior,prompt_type=info_map[concept]
                    exclude_words=exclude_words_per_type[prompt_type]
                    exclude_list=[]
                    for fname in cap_data:
                        caption=cap_data[fname]
                        for word in exclude_words:
                            if word in caption:
                                exclude_list.append(fname)
                                break
                else:
                    exclude_list=None
                # log_path=os.path.join(exp_path,'result.txt')
                if args.exclude_attr_change:
                    if args.grounded:
                        score_name='masked_dino_noattr'
                        fake_root=os.path.join(exp_path,'masked')
                        log_path=os.path.join(exp_path,'result.txt')
                        if not os.path.exists(log_path):
                            continue
                    else:
                        score_name='dino_noattr'
                        fake_root=os.path.join(exp_path,'generated')
                else:
                    if args.grounded:
                        score_name='masked_dino'
                        fake_root=os.path.join(exp_path,'masked')
                        log_path=os.path.join(exp_path,'result.txt')
                        if not os.path.exists(log_path):
                            print('log_path',exp)
                            continue
                    else:
                        score_name='dino'
                        fake_root=os.path.join(exp_path,'generated')
                dst_path=os.path.join(exp_path,'{}.json'.format(score_name))
                if os.path.exists(dst_path) and (not args.ignore_legacy):
                    read_data=json.load(open(dst_path))
                    result_line='{}\t\t{}'.format(exp,read_data[score_name])
                    concept_results.append(result_line)
                    continue
                if not os.path.exists(fake_root):
                    continue
                src_images=[Image.open(os.path.join(real_root,item)).convert('RGB').resize((512,512)) for item in os.listdir(real_root)]
                generated_images=[]
                assert len(os.listdir(fake_root))==200,'gen200'
                for item in os.listdir(fake_root):
                    if not item.endswith(('.png','.jpg')):
                        continue
                    item_name=item.split('.')[0]
                    if exclude_list is not None and item_name in exclude_list:
                        # print(item_name)
                        continue
                    generated_images.append(Image.open(os.path.join(fake_root,item)))
                if not len(generated_images):
                    continue
                score=dino_eval.img_to_img_similarity(src_images=src_images,generated_images=generated_images)
                result_line='{}\t{}'.format(exp,score)
                print(result_line)
                concept_results.append(result_line)
                dst_file=open(os.path.join(exp_path,'{}.json'.format(score_name)),'w')
                dst_data={score_name:float(score)}
                json.dump(dst_data,dst_file)
            if args.sort:
                concept_results = sort_by_mlm_and_s(concept_results)
            if len(concept_results):
                print(concept)
            for item in concept_results:
                print(item)
            print()