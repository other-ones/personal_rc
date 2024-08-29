import time
import numpy as np
import os
from utils import float_to_str
concepts=os.listdir('/data/twkim/diffusion/personalization/collected/images')
info_map={
    'cat1': ('cat','cat'),
    'backpack':('backpack','backpack'),
    'teddybear':('bear','teddybear'),
    'wooden_pot':('pot','wooden_pot'),
    'vase':('vase','vase'),
    'pet_cat1':('cat','pet'),
    'pet_dog1':('dog','pet'),
    'barn': ('barn','building'),
    'chair1': ('chair','chair'),
    'cat_statue': ('toy','toy'),
    'rc_car':('toy','toy'),
    'pink_sunglasses':('sunglasses','sunglasses'),
    'dog3': ('dog','pet'),
    'dog6': ('dog','pet'),
    'flower1':('flower','flower'),

}
lambda_mlm=0.001


import subprocess as sp
def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values




ports=np.arange(5000,6000)
target_devices=[0,1,2,3,4,5,6,7]
stats=get_gpu_memory()
for stat_idx,stat in enumerate(stats):
    if stat>2e4 and stat_idx in target_devices:
        break
device_idx=stat_idx
idx=0
# dirs=['multi','single']
seed=2940
include_prior_concept=1
ppos_list=[0.2,0.1,0.3,0.5,1.0]




if include_prior_concept:
    dir_name='singlev3_prior_seed{}'.format(seed)
else:
    dir_name='singlev3_noprior_seed{}'.format(seed)
dir_path=os.path.join('saved_models/ti_models',dir_name)
log_dir='logs/generate/{}'.format(dir_name)
os.makedirs(log_dir,exist_ok=True)    
delay=30
num_images_per_prompt=8
concepts=os.listdir(dir_path)



    
for cidx,concept in enumerate(info_map.keys()):
    for ppos in ppos_list:
        ppos_str=float_to_str(ppos)
        ppos_str=ppos_str.replace('.','P')
        if concept not in info_map:
            continue
        concept_path=os.path.join(dir_path,concept)
        exps=os.listdir(concept_path)
        for exp_idx,exp in enumerate(exps):
            if '_rev' in exp:
                rev=1
            else:
                rev=0
            if not 'specific2' in exp:
                continue
            # if not ('nomlm' in exp or 'mprob015' in exp):
            #     continue
            prior,category=info_map[concept]
            learned_embed_path1=os.path.join(concept_path,exp,'checkpoints/learned_embeds_s3000.pt')
            if not os.path.exists(learned_embed_path1):
                print(learned_embed_path1,'does not exist')
                continue
            exp_name=exp
            exp_name+='_s3000'
            exp_name+='_ppos{}'.format(ppos_str)

            output_dir=os.path.join('results/{}/{}'.format(dir_name,concept))
            exp_path=os.path.join(output_dir,exp_name)
            if os.path.exists(exp_path):
                print(exp_path,'exists')
                continue
            while True:
                stats=get_gpu_memory()
                found=False
                for stat_idx in target_devices:
                    stat=stats[stat_idx]    
                    if stat>2e4 :
                        device_idx=stat_idx
                        found=True
                        break
                if found:
                    break
                print(exp,'sleep','{}/{}'.format(cidx+1,len(concepts)))
                time.sleep(delay)
            print(exp_name,device_idx)
            log_path=os.path.join(log_dir,exp_name+'.out')
            command='export CUDA_VISIBLE_DEVICES={};'.format(device_idx)
            command+='export CUBLAS_WORKSPACE_CONFIG=:4096:8;'
            command+='accelerate launch --main_process_port {} generate_single.py \\\n'.format(ports[idx],idx)
            command+='--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \\\n'
            command+='--train_data_dir1="/data/twkim/diffusion/personalization/collected/images/{}" \\\n'.format(concept)
            command+='--placeholder_token1="<{}>" \\\n'.format(concept)
            command+='--prior_concept1="{}" \\\n'.format(prior)
            command+='--resolution=512 \\\n'
            command+='--dst_exp_path={} \\\n'.format(exp_path)
            command+='--eval_batch_size=15 \\\n'
            command+='--num_images_per_prompt={} \\\n'.format(num_images_per_prompt)
            command+='--rev={} \\\n'.format(rev)
            command+='--output_dir="{}" \\\n'.format(output_dir)
            command+='--seed={} \\\n'.format(seed)
            command+='--mask_tokens="[MASK]" \\\n'
            command+='--learned_embed_path1="{}" \\\n'.format(learned_embed_path1)
            command+='--calibrate_ppos1="{}" \\\n'.format(ppos)
            command+='--prompt_type="{}" \\\n'.format(category)
            command+='--include_prior_concept={} > {} 2>&1 &'.format(include_prior_concept,log_path)
            os.system(command)
            print('STARTED')
            idx+=1
            time.sleep(15)




