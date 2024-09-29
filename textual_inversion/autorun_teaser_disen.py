from utils import float_to_str
import time
import numpy as np
import os
import socket
hostname = socket.gethostname()
print(hostname,'hostname')
# concepts=os.listdir('/data/twkim/diffusion/personalization/collected/images')
info_map_03={
    # train_prior/eval_prior/train_prompt_type/eval_prompt_type
    # 'man_dog1':('duck','duck toy','nonliving','nonliving'),
    # 'chair1': ('chair','chair','nonliving','nonliving'),
    # 'dog6': ('dog','dog','pet','living'),
    'man_dog1_dog':('dog','dog','car','car'),
    'boy_car_car':('car','car','car','car'),

    # 'cat1': ('cat','cat','pet','living'),
    # 'pet_cat1':('cat','cat','pet','living'),
    # 'wooden_pot':('pot','wooden pot','nonliving','nonliving'),
    # 'backpack_dog':('backpack','backpack','nonliving','nonliving'),
    # 'poop_emoji':('toy','toy','nonliving','nonliving'),
    # 'cat2':('cat','cat','pet','living'),
    # 'dog3':  ('dog','dog','pet','living'),
    # 'pet_dog1':('dog','dog','pet','living'),
    # 'backpack':('backpack','backpack','nonliving','nonliving'),
    # 'teddybear':('teddy','teddy bear','nonliving','nonliving'),
    # 'cat_statue': ('toy','toy','nonliving','nonliving'),
    # 'rc_car':('toy','toy','nonliving','nonliving'),


    # NOT USED
    # 'red_cartoon':('character','cartoon character','pet','living'),
    # 'candle':('candle','candle','nonliving','nonliving'),
    # 'can':('can','can','nonliving','nonliving'),
    # 'pink_sunglasses':('sunglasses','sunglasses'),
    # 'barn': ('barn','barn'),
    # 'flower1':('flower','flower'),
}

if '03' in hostname:
    info_map=info_map_03
    host_suffix='03'
elif 'ubuntu' in hostname:
    info_map=info_map_01
    host_suffix='01'
elif '07' in hostname:
    info_map=info_map_03
    host_suffix='07'
elif '04' in hostname:
    info_map=info_map_03
    host_suffix='04'
else:
    assert False
concepts=list(info_map.keys())

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
np.random.shuffle(ports)
target_devices=[0,1]
resume_seed=7777
gen_seed=6804
include_prior=1
delay=25
benchmark='custom_disen'
dir_name='custom'
exclude_cap_types=None
print('GENERATION')
# GENERATION
dir_path=os.path.join('saved_models/ti_models',dir_name)
delay=30
num_images_per_prompt=8
port_idx=0
for gen_target_step in [2000]:
    for cidx,concept in enumerate(concepts):
        if concept not in info_map:
            continue
        concept_path=os.path.join(dir_path,concept)
        image_concept='_'.join(concept.split('_')[:-1])
        if not os.path.exists(concept_path):
            continue
        exps=os.listdir(concept_path)
        for exp_idx,exp in enumerate(exps):
            train_prior,eval_prior,train_prompt_type,eval_prompt_type=info_map[concept]
            learned_embed_path1=os.path.join(concept_path,exp,'checkpoints/learned_embeds_s{}.pt'.format(gen_target_step))
            if not os.path.exists(learned_embed_path1):
                print(learned_embed_path1,'does not exist')
                continue
            exp_name=exp
            exp_name+='_s{}'.format(gen_target_step)
            output_dir=os.path.join(f'results/teasers_custom/{benchmark}_seed{gen_seed}/{dir_name}/{concept}')
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
                print('SLEEP GENEARTING',exp,'sleep','{}/{}'.format(cidx+1,len(concepts)))
                time.sleep(delay)
            print(f'{benchmark}\t{output_dir}\t{device_idx}')
            os.makedirs(exp_path,exist_ok=True)   
            log_path=os.path.join(exp_path,'log.out')
            command='export CUDA_VISIBLE_DEVICES={};'.format(device_idx)
            command+='export CUBLAS_WORKSPACE_CONFIG=:4096:8;'
            command+='accelerate launch --main_process_port {} ti_generate_teaser.py \\\n'.format(ports[port_idx])
            command+='--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \\\n'
            command+='--train_data_dir1="/data/twkim/diffusion/personalization/custom_images/{}" \\\n'.format(image_concept)
            command+='--placeholder_token1="<{}>" \\\n'.format(concept)
            command+='--train_prior_concept1="{}" \\\n'.format(train_prior)
            command+='--eval_prior_concept1="{}" \\\n'.format(eval_prior)
            command+='--resolution=512 \\\n'
            command+='--dst_exp_path={} \\\n'.format(exp_path)
            command+='--benchmark_path="../datasets_pkgs/eval_prompts/{}.json" \\\n'.format(benchmark)
            command+='--eval_batch_size=15 \\\n'
            command+='--num_images_per_prompt={} \\\n'.format(num_images_per_prompt)
            command+='--output_dir="{}" \\\n'.format(output_dir)
            command+='--seed={} \\\n'.format(gen_seed)
            command+='--mask_tokens="[MASK]" \\\n'
            command+='--learned_embed_path1="{}" \\\n'.format(learned_embed_path1)
            command+='--calibrate_ppos1=0 \\\n'
            command+='--train_prompt_type="{}" \\\n'.format(train_prompt_type)
            command+='--eval_prompt_type="{}" \\\n'.format(eval_prompt_type)
            command+='--include_prior_concept={} > {} 2>&1 &'.format(include_prior,log_path)
            os.system(command)
            print('GENERATION STARTED')
            port_idx+=1
            time.sleep(delay)