import time
import numpy as np
import os
import socket
hostname = socket.gethostname()
print(hostname,'hostname')
concepts=os.listdir('/data/twkim/diffusion/personalization/collected/images')
info_map_03={
    'backpack':('backpack','nonliving'),
    'teddybear':('teddybear','nonliving'),
    'wooden_pot':('pot','nonliving'),
    'vase':('vase','nonliving'),
    'cat1': ('cat','pet'),
    'pet_cat1':('cat','pet'),
    'pet_dog1':('dog','pet'),
    'barn': ('barn','building'),
    'chair1': ('chair','nonliving'),
    'cat_statue': ('toy','nonliving'),

    # cat_statue  chair1  dog3  pink_sunglasses  rc_car  wooden_pot
    # 'pink_sunglasses':('sunglasses','sunglasses'),
    # 'rc_car':('toy','nonliving'),
    # 'dog3': ('dog','pet'),
    # 'dog6': ('dog','pet'),
    # 'flower1':('flower','flower'),
}
info_map_01={
    # 'backpack':('backpack','nonliving'),
    # 'teddybear':('teddybear','nonliving'),
    # 'wooden_pot':('pot','nonliving'),
    # 'vase':('vase','nonliving'),
    # 'cat1': ('cat','pet'),
    # 'pet_cat1':('cat','pet'),
    # 'pet_dog1':('dog','pet'),
    # 'barn': ('barn','building'),
    # 'chair1': ('chair','nonliving'),
    # 'cat_statue': ('toy','nonliving'),
    
    # 'pink_sunglasses':('sunglasses','sunglasses'),
    'rc_car':('toy','nonliving'),
    'dog3': ('dog','pet'),
    'dog6': ('dog','pet'),
    # 'flower1':('flower','flower'),
}
if '03' in hostname:
    info_map=info_map_03
    delay=15
elif 'ubuntu' in hostname:
    info_map=info_map_01
    delay=30
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
# dirs=['multi','single']
dirs=['single']
for dir in dirs:
    dir_path=os.path.join('saved_models/pplus_models',dir)
    log_dir='logs/pplus/generate/{}'.format(dir)
    os.makedirs(log_dir,exist_ok=True)    
    concepts=os.listdir(dir_path)
    for concept in concepts:
        if concept not in info_map:
            continue
        concept_path=os.path.join(dir_path,concept)
        exps=os.listdir(concept_path)
        for exp_idx,exp in enumerate(exps):
            prior,category=info_map[concept]
            learned_embed_path1=os.path.join(concept_path,exp,'checkpoints/learned_embeds_s3000.pt')
            if not os.path.exists(learned_embed_path1):
                print(learned_embed_path1,'does not exist')
                continue
            exp_name=learned_embed_path1.split('/')[-3]
            if 'noprior' in exp_name:
                include_prior_concept=0
            else:
                assert '_prior_' in exp_name
                include_prior_concept=1
            output_dir=os.path.join('results/pplus_results/{}/{}'.format(dir,concept))
            exp_path=os.path.join(output_dir,exp_name)
            print(exp_path,'exp_path')
            if os.path.exists(exp_path):
                print(exp_name,'exists')
                continue
            while True:
                stats=get_gpu_memory()
                stat=stats[stat_idx%len(stats)]
                if stat>2e4:
                    device_idx=stat_idx
                    stat_idx+=1
                    break
                print('sleep waiting for {}'.format(exp_name),'GPU[{}] is busy FREE: {}MB'.format(stat_idx,stat),'# Remaining Exps: {}'.format(len(exps)-exp_idx))
                time.sleep(delay)
                stat_idx+=1
                stat_idx=(stat_idx%len(stats))
            print(exp_name,device_idx)
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
            command+='--seed=1234 \\\n'
            command+='--num_vectors1=9 \\\n'
            command+='--mask_tokens="[MASK]" \\\n'
            command+='--learned_embed_path1="{}" \\\n'.format(learned_embed_path1)
            command+='--prompt_type="{}" \\\n'.format(category)

            command+='--include_prior_concept={} > {} 2>&1 &'.format(include_prior_concept,log_path)
            os.system(command)
            print('STARTED')
            idx+=1
            time.sleep(delay)

    


