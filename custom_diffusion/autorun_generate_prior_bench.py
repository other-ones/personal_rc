from utils import float_to_str,invert_scientific_notation
import time
import numpy as np
import os
import socket
hostname = socket.gethostname()
print(hostname,'hostname')
concepts=os.listdir('/data/twkim/diffusion/personalization/collected/images')

info_map={
    # train_prior/eval_prior/train_prompt_type/eval_prompt_type
    'pet_cat1':('cat','cat','pet','living'),
    'cat_statue': ('toy','toy','nonliving','nonliving'),
    'backpack_dog':('backpack','backpack','nonliving','nonliving'),
    'rc_car':('toy','toy','nonliving','nonliving'),
    'cat1': ('cat','cat','pet','living'),
    'backpack':('backpack','backpack','nonliving','nonliving'),
    
    'teapot':('teapot','teapot','nonliving','nonliving'),
    'chair1': ('chair','chair','nonliving','nonliving'),
    'dog6': ('dog','dog','pet','living'),
    'duck_toy':('duck','duck toy','nonliving','nonliving'),

    'dog3':  ('dog','dog','pet','living'),
    'cat2':('cat','cat','pet','living'),

    'wooden_pot':('pot','wooden pot','nonliving','nonliving'),
    'poop_emoji':('toy','toy','nonliving','nonliving'),
    'pet_dog1':('dog','dog','pet','living'),
    'teddybear':('teddy','teddy bear','nonliving','nonliving'),
}

if '03' in hostname:
    target_devices=[0,1,2,3,4,5,6,7]
    host_suffix='03'
elif 'ubuntu' in hostname:
    target_devices=[0,1]
    host_suffix='01'
elif '07' in hostname:
    target_devices=[0,1,2]
    host_suffix='07'
elif '04' in hostname:
    target_devices=[2,3,4,5,6,7]
    host_suffix='04'
    info_map=info_map
else:
    assert False

import subprocess as sp
def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

stats=get_gpu_memory()
for stat_idx,stat in enumerate(stats):
    if stat>2e4:
        break

ports=np.arange(1111,2222)
seed=2940
dir_name='sd1'
num_devices=1
port_idx=0
num_images_per_prompt=8
for concept_idx,concept in enumerate(list(info_map.keys())):
    train_prior,eval_prior,train_prompt_type,eval_prompt_type=info_map[concept]
    eval_prior_split=eval_prior.split()
    run_name='_'.join(eval_prior_split)
    output_dir=os.path.join('results/priors/{}'.format(dir_name))
    dst_exp_path=os.path.join(output_dir,run_name)
    if os.path.exists(dst_exp_path):
        print(dst_exp_path,'exists')
        continue
    while True:
        stats=get_gpu_memory()
        found=False
        available_devices=[]
        for stat_idx in target_devices:
            stat=stats[stat_idx]    
            if stat>2e4 :
                available_devices.append(str(stat_idx))
        if len(available_devices)>=num_devices:
            break
        print(f'SLEEP PRIOR GENERATION\t{dir_name}\t{run_name}\t{concept_idx+1}/{len(list(info_map.keys()))}')
        time.sleep(10)
    device_idxs=','.join(available_devices[:num_devices])
    print(f"{dir_name}\t{run_name}\tDEVICE:{device_idxs}")
    os.makedirs(dst_exp_path,exist_ok=True) 
    log_path=os.path.join(dst_exp_path,'log.out')
    command='export CUDA_VISIBLE_DEVICES={};'.format(device_idxs)
    command+='export CUBLAS_WORKSPACE_CONFIG=:4096:8;'
    command+='accelerate launch --main_process_port {} generate_prior_bench.py \\\n'.format(ports[port_idx])
    command+='--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \\\n'
    command+='--resolution=512 \\\n'
    command+='--eval_batch_size=15 \\\n'
    command+='--num_images_per_prompt={} \\\n'.format(num_images_per_prompt)
    command+='--seed={} \\\n'.format(seed)
    command+='--dst_exp_path="{}" \\\n'.format(dst_exp_path)
    command+='--eval_prior_concept1="{}" \\\n'.format(eval_prior)
    command+='--eval_prompt_type="{}" \\\n'.format(eval_prompt_type)
    command+='--benchmark_path="../datasets_pkgs/eval_prompts/dreambooth.json" \\\n'
    command+='--include_prior_concept=1 > {} 2>&1 &'.format(log_path)
    os.system(command)
    print('GENERATION STARTED')
    port_idx+=1
    time.sleep(30)
