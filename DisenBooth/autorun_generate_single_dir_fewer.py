from utils import float_to_str
import time
import numpy as np
import os
import socket
hostname = socket.gethostname()
print(hostname,'hostname')
concepts=os.listdir('/data/twkim/diffusion/personalization/collected/images')
info_map_03={
    # qlab03
    'dog6': ('dog','pet'),
    'pet_cat1':('cat','pet'),
    'wooden_pot':('pot','nonliving'),
    'vase':('vase','nonliving'),
    # 'pet_dog1':('dog','pet'),
    # 'dog3': ('dog','pet'),
    # 'backpack':('backpack','nonliving'),
    # 'cat1': ('cat','pet'),
    # 'barn': ('barn','building'),
    # 'chair1': ('chair','nonliving'),

    # qlab01
    # 'cat_statue': ('toy','nonliving'),
    # 'rc_car':('toy','nonliving'),
    # 'teddybear':('bear','nonliving'),
    # 'pink_sunglasses':('sunglasses','sunglasses'),
}
info_map_01={
    # qlab03
    # 'pet_dog1':('dog','pet'),
    # 'pet_cat1':('cat','pet'),
    # 'dog3': ('dog','pet'),
    # 'dog6': ('dog','pet'),
    # 'backpack':('backpack','nonliving'),
    # 'vase':('vase','nonliving'),
    # 'cat1': ('cat','pet'),
    # 'barn': ('barn','building'),
    # 'wooden_pot':('pot','nonliving'),
    # 'chair1': ('chair','nonliving'),

    # qlab01
    'teddybear':('bear','nonliving'),
    'rc_car':('toy','nonliving'),
    # 'cat_statue': ('toy','nonliving'),
    # 'pink_sunglasses':('sunglasses','sunglasses'),
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




ports=np.arange(5000,6000)
stats=get_gpu_memory()
for stat_idx,stat in enumerate(stats):
    if stat>2e4:
        break
device_idx=stat_idx
idx=0
if '03' in hostname:
    delay=30
    # target_devices=[0,1,2,3,4,5,6,7]
    target_devices=[0,1,2,3,4,5,6,7]
else:
    delay=45
    target_devices=[0,1]
dir_path=os.path.join('saved_models/disenbooth_models/fewer/single')
log_dir='logs/disenbooth/generate/fewer/single'
os.makedirs(log_dir,exist_ok=True)    
concepts=os.listdir(dir_path)
concepts=sorted(concepts)
target_steps=[2500,2000]
for target_step in target_steps:
    for concept in info_map.keys():
        if concept not in info_map:
            continue
        concept_path=os.path.join(dir_path,concept)
        if not os.path.exists(concept_path):
            continue
        exps=os.listdir(concept_path)
        for exp_idx,exp in enumerate(exps):
            prior,category=info_map[concept]
            if 'nomlm' in exp:
                learned_embed_path1=None
            else:
                learned_embed_path1=os.path.join(concept_path,exp,'checkpoints/checkpoint-{}/learned_embeds_{}.pt'.format(target_step,target_step))
            resume_unet_path=os.path.join(concept_path,exp,'checkpoints/checkpoint-{}/unet_{}.pt'.format(target_step,target_step))
            if not os.path.exists(resume_unet_path):
                print(resume_unet_path,'does not exist')
                continue
            # exp_name=resume_unet_path.split('/')[-4]
            output_dir=os.path.join('results/disenbooth/fewer/single/{}'.format(concept))
            dst_exp=exp+'_s{}'.format(target_step)
            exp_dir=os.path.join(output_dir,dst_exp)
            if os.path.exists(exp_dir):
                print(dst_exp,'exists')
                continue
            while True:
                stats=get_gpu_memory()
                stat=stats[stat_idx%len(stats)]
                found=False
                for stat_idx in target_devices:
                    stat=stats[stat_idx]    
                    if stat>2e4:
                        device_idx=stat_idx
                        found=True
                        break
                    # time.sleep(5)
                if found:
                    break
                print('sleep waiting for {}'.format(dst_exp))
                time.sleep(delay)
                stat_idx+=1
                stat_idx=(stat_idx%len(stats))
            print(dst_exp,device_idx)
            log_path=os.path.join(log_dir,dst_exp+'.out')
            command='export CUDA_VISIBLE_DEVICES={};'.format(device_idx)
            command+='accelerate launch --main_process_port {} generate_single.py \\\n'.format(ports[idx],idx)
            command+='--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \\\n'
            command+='--train_data_dir1="/data/twkim/diffusion/personalization/collected/images/{}" \\\n'.format(concept)
            command+='--placeholder_token1="<{}>" \\\n'.format(concept)
            command+='--prior_concept1="{}" \\\n'.format(prior)
            command+='--resolution=512 \\\n'
            command+='--exp_dir={} \\\n'.format(exp_dir)
            command+='--eval_batch_size=16 \\\n'
            command+='--num_images_per_prompt=15 \\\n'
            command+='--output_dir="{}" \\\n'.format(output_dir)
            command+='--seed=1234 \\\n'
            command+='--mask_tokens="[MASK]" \\\n'
            command+='--resume_unet_path="{}" \\\n'.format(resume_unet_path)
            command+='--learned_embed_path1="{}" \\\n'.format(learned_embed_path1)
            command+='--prompt_type="{}" \\\n'.format(category)
            command+='--include_prior_concept=1 > {} 2>&1 &'.format(log_path)
            os.system(command)
            print('STARTED, sleeping..')
            idx+=1
            time.sleep(delay)
            print('woke up')

        


