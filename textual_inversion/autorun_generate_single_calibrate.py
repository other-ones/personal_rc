from utils import float_to_str
import time
import numpy as np
import os
import socket
hostname = socket.gethostname()
print(hostname,'hostname')
concepts=os.listdir('/data/twkim/diffusion/personalization/collected/images')
info_map_03={
    'backpack':('backpack','nonliving'),
    'pet_cat1':('cat','pet'),
    'pet_dog1':('dog','pet'),
    'vase':('vase','nonliving'),
    # 'teddybear':('teddybear','nonliving'),
    # 'dog6': ('dog','pet'),
    # 'cat1': ('cat','pet'),
    # 'barn': ('barn','building'),
    # 'wooden_pot':('pot','nonliving'),

    # 'dog3': ('dog','pet'),
    # 'chair1': ('chair','nonliving'),
    # 'cat_statue': ('toy','nonliving'),
    # 'rc_car':('toy','nonliving'),
    # 'pink_sunglasses':('sunglasses','sunglasses'),
    # 'flower1':('flower','flower'),
}
info_map_01={
    # 'backpack':('backpack','nonliving'),
    # 'pet_cat1':('cat','pet'),
    # 'pet_dog1':('dog','pet'),
    # 'vase':('vase','nonliving'),
    'teddybear':('teddybear','nonliving'),
    'dog6': ('dog','pet'),
    # 'cat1': ('cat','pet'),
    # 'barn': ('barn','building'),
    # 'wooden_pot':('pot','nonliving'),

    # 'dog3': ('dog','pet'),
    # 'chair1': ('chair','nonliving'),
    # 'cat_statue': ('toy','nonliving'),
    # 'rc_car':('toy','nonliving'),
    # 'pink_sunglasses':('sunglasses','sunglasses'),
    # 'flower1':('flower','flower'),
}

if '03' in hostname:
    info_map=info_map_03
elif 'ubuntu' in hostname:
    info_map=info_map_01
lambda_mlm=0.001


import subprocess as sp
def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


log_dir='logs/ti_models/generate/calibrate/single_prior'
os.makedirs(log_dir,exist_ok=True)    


ports=np.arange(5000,6000)
stats=get_gpu_memory()
for stat_idx,stat in enumerate(stats):
    if stat>2e4:
        break
device_idx=stat_idx

target_norm=0
include_prior_concept=1
pos_values=[0,0.1,1,10]
lambda_mlms=[0,0.001]
for lambda_mlm in lambda_mlms:
    lambda_mlm_str=float_to_str(lambda_mlm).replace('.','')
    for pos_value in pos_values:
        pos_value_str=float_to_str(pos_value).replace('.','')
        for idx,concept in enumerate(list(info_map.keys())):
            prior,category=info_map[concept]
            if include_prior_concept:
                if lambda_mlm:
                    learned_embed_path1='saved_models/ti_models/single_prior/{}/ti_norm{}_prior_mlm{}_{}/checkpoints/learned_embeds_s3000.pt'.format(concept,target_norm,lambda_mlm_str,concept)
                else:
                    learned_embed_path1='saved_models/ti_models/single_prior/{}/ti_norm{}_prior_nomlm_{}/checkpoints/learned_embeds_s3000.pt'.format(concept,target_norm,concept)
            else:
                if lambda_mlm:
                    learned_embed_path1='saved_models/ti_models/single/{}/ti_norm{}_noprior_mlm{}_{}/checkpoints/learned_embeds_s3000.pt'.format(concept,target_norm,lambda_mlm_str,concept)
                else:
                    learned_embed_path1='saved_models/ti_models/single/{}/ti_norm{}_noprior_nomlm_{}/checkpoints/learned_embeds_s3000.pt'.format(concept,target_norm,concept)
            if not os.path.exists(learned_embed_path1):
                print(learned_embed_path1,'does not exists')
                continue
            
            exp_name=learned_embed_path1.split('/')[-3]
            output_dir=os.path.join('results/ti_results/single_prior/calibrate/{}'.format(concept))
            dst_exp_path=os.path.join(output_dir,exp_name)
            dst_exp_path+='_kpos1_{}_ppos1_{}'.format(pos_value_str,pos_value_str)
            if os.path.exists(dst_exp_path):
                print(dst_exp_path,'exists')
                continue
            while True:
                stats=get_gpu_memory()
                stat=stats[stat_idx%len(stats)]
                if stat>2e4:
                    device_idx=stat_idx
                    stat_idx+=1
                    break
                print('sleep waiting for {}'.format(exp_name),'GPU[{}] is busy FREE: {}MB'.format(stat_idx,stat),'# Remaining Exps: {}'.format(len(info_map)-idx))
                time.sleep(10)
                stat_idx+=1
                stat_idx=(stat_idx%len(stats))
            print(dst_exp_path,device_idx)
            log_path=os.path.join(log_dir,exp_name+'.out')
            command='export CUDA_VISIBLE_DEVICES={};'.format(device_idx)
            command+='accelerate launch --main_process_port {} generate_single.py \\\n'.format(ports[idx],idx)
            command+='--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \\\n'
            command+='--train_data_dir1="/data/twkim/diffusion/personalization/collected/images/{}" \\\n'.format(concept)
            command+='--placeholder_token1="<{}>" \\\n'.format(concept)
            command+='--prior_concept1="{}" \\\n'.format(prior)
            command+='--resolution=512 \\\n'
            command+='--eval_batch_size=18 \\\n'
            command+='--num_images_per_prompt=15 \\\n'
            command+='--output_dir="{}" \\\n'.format(output_dir)
            command+='--dst_exp_path="{}" \\\n'.format(dst_exp_path)
            command+='--seed=1234 \\\n'
            command+='--calibrate_kpos1={} \\\n'.format(pos_value)
            command+='--calibrate_ppos1={} \\\n'.format(pos_value)
            command+='--mask_tokens="[MASK]" \\\n'
            command+='--learned_embed_path1="{}" \\\n'.format(learned_embed_path1)
            command+='--prompt_type="{}" \\\n'.format(category)
            command+='--include_prior_concept={} > {} 2>&1 &'.format(include_prior_concept,log_path)
            
            os.system(command)
            print('STARTED')
            time.sleep(20)
        


