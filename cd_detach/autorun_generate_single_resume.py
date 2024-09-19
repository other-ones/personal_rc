from utils import float_to_str
import time
import numpy as np
import os
import socket
hostname = socket.gethostname()
concepts=os.listdir('/data/twkim/diffusion/personalization/collected/images')
info_map_03={
    # qlab03
    # 'dog6': ('dog','pet'),
    # 'vase':('vase','nonliving'),
    # 'pet_cat1':('cat','pet'),
    # 'wooden_pot':('pot','nonliving'),
    # 'pet_dog1':('dog','pet'),
    'dog3': ('dog','pet'),
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
    'dog6': ('dog','pet'),
    # 'backpack':('backpack','nonliving'),
    # 'vase':('vase','nonliving'),
    # 'cat1': ('cat','pet'),
    # 'barn': ('barn','building'),
    # 'chair1': ('chair','nonliving'),

    # qlab01
    # 'teddybear':('bear','nonliving'),
    # 'wooden_pot':('pot','nonliving'),
    # 'rc_car':('toy','nonliving'),
    # 'cat_statue': ('toy','nonliving'),
    # 'pink_sunglasses':('sunglasses','sunglasses'),
}
if '03' in hostname:
    info_map=info_map_03
    delay=25
elif 'ubuntu' in hostname:
    info_map=info_map_01
    delay=40


import subprocess as sp
def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values



target_step=250
ports=np.arange(5000,6000)
np.random.shuffle(ports)
stats=get_gpu_memory()
for stat_idx,stat in enumerate(stats):
    if stat>2e4:
        break
device_idx=stat_idx
port_idx=0

for seed in [8881,2940]:
    dir_name='single_seed{}'.format(seed)
    num_devices=1
    target_devices=[0,1,2,3,4,5,6,7]
    for target_step in [500,1000,1500]:
        dir_path=os.path.join('saved_models/custom_diffusion',dir_name)
        log_dir='logs/generate/{}'.format(dir_name)
        os.makedirs(log_dir,exist_ok=True)    
        concepts=os.listdir(dir_path)
        concepts=sorted(concepts)
        for concept in concepts:
            if concept not in info_map:
                continue
            concept_path=os.path.join(dir_path,concept)
            exps=os.listdir(concept_path)
            for exp_idx,exp in enumerate(exps):
                if not '_resume' in exp:
                    continue
                prior,category=info_map[concept]
                learned_embed_path1=os.path.join(concept_path,exp,'checkpoints/checkpoint-{}'.format(target_step))
                if not os.path.exists(learned_embed_path1):
                    print(learned_embed_path1,'does not exist')
                    continue
                resume_exp=exp.split('_lr')[0]
                resume_step=exp.split('_resume')[-1]
                resume_path=os.path.join("saved_models/custom_diffusion/single_seed{}/{}/{}/checkpoints/checkpoint-{}".format(seed,concept,resume_exp,resume_step))
                if not os.path.exists(resume_path):
                    print(resume_path,'resume path not exist')
                    continue
                exp_name=exp
                exp_name+='_s{}'.format(target_step)
                output_dir=os.path.join('results/{}/{}'.format(dir_name,concept))
                exp_path=os.path.join(output_dir,exp_name)
                if os.path.exists(exp_path):
                    print(exp_path,'exists')
                    continue
                while True:
                    idle_devices=[]
                    stats=get_gpu_memory()
                    for device_idx in target_devices:
                        stat=stats[device_idx]   
                        if stat>2e4:
                            idle_devices.append(str(device_idx))
                    if len(idle_devices)>=num_devices:
                        break
                    print('sleep waiting for {}'.format(exp_name))
                    time.sleep(delay)
                    stat_idx+=1
                    stat_idx=(stat_idx%len(stats))
                log_path=os.path.join(log_dir,exp_name+'.out')
                idle_devices=idle_devices[:num_devices]
                running_device=','.join(idle_devices)
                print(exp_path,running_device)
                command='export CUDA_VISIBLE_DEVICES={};'.format(running_device)
                command+='accelerate launch --main_process_port {} generate_single.py \\\n'.format(ports[port_idx])
                command+='--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \\\n'
                command+='--train_data_dir1="/data/twkim/diffusion/personalization/collected/images/{}" \\\n'.format(concept)
                command+='--placeholder_token1="<{}>" \\\n'.format(concept)
                command+='--prior_concept1="{}" \\\n'.format(prior)
                command+='--resolution=512 \\\n'
                command+='--eval_batch_size=15 \\\n'
                command+='--num_images_per_prompt=15 \\\n'
                command+='--learned_embed_path1="{}" \\\n'.format(learned_embed_path1)
                command+='--output_dir="{}" \\\n'.format(output_dir)
                command+='--seed={} \\\n'.format(seed)
                command+='--mask_tokens="[MASK]" \\\n'
                command+='--resume_path="{}" \\\n'.format(resume_path)
                command+='--prompt_type="{}" \\\n'.format(category)
                command+='--include_prior_concept=1 > {} 2>&1 &'.format(log_path)
                os.system(command)
                print('STARTED')
                port_idx+=1
                time.sleep(delay)

    


