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
    # 'duck_toy':('duck','duck toy','nonliving','nonliving'),
    # 'dog6': ('dog','dog','pet','living'),
    'teapot':('teapot','teapot','nonliving','nonliving'),
    # 'pet_cat1':('cat','cat','pet','living'),

    # 'cat1': ('cat','cat','pet','living'),
    # 'wooden_pot':('pot','wooden pot','nonliving','nonliving'),
    # 'backpack_dog':('backpack','backpack','nonliving','nonliving'),
    # 'poop_emoji':('toy','toy','nonliving','nonliving'),
    # 'cat2':('cat','cat','pet','living'),
    # 'dog3':  ('dog','dog','pet','living'),
    # 'pet_dog1':('dog','dog','pet','living'),

    # 'backpack':('backpack','backpack','nonliving','nonliving'),
    # 'cat_statue': ('toy','toy','nonliving','nonliving'),
    # 'rc_car':('toy','toy','nonliving','nonliving'),
    # 'chair1': ('chair','chair','nonliving','nonliving'),
    # 'teddybear':('teddy','teddy bear','nonliving','nonliving'),

    # 'red_cartoon':('character','cartoon character','pet','living'),
    # 'candle':('candle','candle','nonliving','nonliving'),
    # 'can':('can','can','nonliving','nonliving'),
    # 'pink_sunglasses':('sunglasses','sunglasses'),
    # 'barn': ('barn','barn'),
    # 'flower1':('flower','flower'),
}
info_map_01={
    # train_prior/eval_prior/train_prompt_type/eval_prompt_type
    'teapot':('teapot','teapot','nonliving','nonliving'),
    'dog6': ('dog','dog','pet','living'),
    'duck_toy':('duck','duck toy','nonliving','nonliving'),
    'pet_cat1':('cat','cat','pet','living'),

    'wooden_pot':('pot','wooden pot','nonliving','nonliving'),
    'backpack_dog':('backpack','backpack','nonliving','nonliving'),
    'poop_emoji':('toy','toy','nonliving','nonliving'),
    'cat2':('cat','cat','pet','living'),
    'cat1': ('cat','cat','pet','living'),
    'dog3':  ('dog','dog','pet','living'),
    'pet_dog1':('dog','dog','pet','living'),
    'backpack':('backpack','backpack','nonliving','nonliving'),
    'teddybear':('bear','teddy bear','nonliving','nonliving'),
    'cat_statue': ('toy','toy','nonliving','nonliving'),
    'rc_car':('toy','toy','nonliving','nonliving'),
    'chair1': ('chair','chair','nonliving','nonliving'),

    # 'red_cartoon':('character','cartoon character','pet','living'),
    # 'candle':('candle','candle','nonliving','nonliving'),
    # 'can':('can','can','nonliving','nonliving'),
    # 'pink_sunglasses':('sunglasses','sunglasses'),
    # 'barn': ('barn','barn'),
    # 'flower1':('flower','flower'),
}
if '03' in hostname:
    target_devices=[0,1]
    host_suffix='03'

elif '04' in hostname:
    target_devices=[0,6,7]
    host_suffix='04'
elif '07' in hostname:
    target_devices=[0,1,2]
    host_suffix='07'

masked_loss=0


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


num_devices=0
while True:
    stats=get_gpu_memory()
    found=False
    available_devices=[]
    for stat_idx in target_devices:
        stat=stats[stat_idx]    
        if stat>2e4 :
            available_devices.append(stat_idx)
    if len(available_devices)>=num_devices:
        break
    print('waiting..')
    time.sleep(30)


ports=np.arange(1111,2222)
fixte_list=[0]
mask_prob_list=[0.15]
resume_seed=7777
gen_seed=6804
rep_id=1
dir_name=f'bigger_seed{resume_seed}_qlab{host_suffix}_rep{rep_id}'
log_dir='logs/train/{}'.format(dir_name)
os.makedirs(log_dir,exist_ok=True)   
# for port_idx,concept in enumerate(list(info_map.keys())):
lr_list=[1e-6]
mlm_batch_size=25
port_idx=0





print('GENERATION')
# GENERATION
dir_path=os.path.join('saved_models/db_models',dir_name)
delay=30
num_images_per_prompt=20
port_idx=0
include_prior_concept=1
ppos_list=[0]
benchmark='teaser4'
concepts=list(info_map.keys())
concepts=sorted(concepts)
for gen_target_step in [1000]:
    for concept_idx,concept in enumerate(concepts):
        if concept not in info_map:
            continue
        concept_path=os.path.join(dir_path,concept)
        if not os.path.exists(concept_path):
            continue
        exps=os.listdir(concept_path)
        for exp_idx,exp in enumerate(exps):
            if '_ti' in exp:
                continue
            if not (('nomlm'in exp) or ('mlm00005' in exp)) :
                continue
            train_prior,eval_prior,train_prompt_type,eval_prompt_type=info_map[concept]
            resume_unet_path=os.path.join(concept_path,exp,'checkpoints/checkpoint-{}/unet_s{:04d}.pt'.format(gen_target_step,gen_target_step))
            resume_text_encoder_path=os.path.join(concept_path,exp,'checkpoints/checkpoint-{}/text_encoder_s{:04d}.pt'.format(gen_target_step,gen_target_step))
            if not os.path.exists(resume_unet_path):
                print(resume_unet_path,'does not exist')
                continue
            exp_name=resume_unet_path.split('/')[-4]
            exp_name+=f'_s{gen_target_step}'
            output_dir=os.path.join('results/teasers/{}_seed{}/db_results/{}/{}'.format(benchmark,gen_seed,dir_name,concept))
            dst_exp_path=os.path.join(output_dir,exp_name)
            if os.path.exists(dst_exp_path):
                print(dst_exp_path,'exists')
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
                # print(exp_name,'sleep',stat_idx,stat)
                print(f"SLEEP GENERATION\t{benchmark}\t{exp_name}")
                time.sleep(10)
            print(f'{benchmark}\t{output_dir}\t{device_idx}')
            os.makedirs(dst_exp_path,exist_ok=True)
            log_path=os.path.join(dst_exp_path,'log.out')
            command='export CUDA_VISIBLE_DEVICES={};'.format(device_idx)
            command+='export CUBLAS_WORKSPACE_CONFIG=:4096:8;'
            command+='accelerate launch --main_process_port {} db_generate_teaser.py \\\n'.format(ports[port_idx])
            command+='--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \\\n'
            command+='--train_data_dir1="/data/twkim/diffusion/personalization/collected/images/{}" \\\n'.format(concept)
            command+='--placeholder_token1="<{}>" \\\n'.format(concept)
            command+='--resolution=512 \\\n'
            command+='--eval_batch_size=15 \\\n'
            command+='--num_images_per_prompt={} \\\n'.format(num_images_per_prompt)
            command+='--output_dir="{}" \\\n'.format(output_dir)
            command+='--seed={} \\\n'.format(gen_seed)
            command+='--mask_tokens="[MASK]" \\\n'
            command+='--resume_unet_path="{}" \\\n'.format(resume_unet_path)
            command+='--resume_text_encoder_path="{}" \\\n'.format(resume_text_encoder_path)
            command+='--dst_exp_path="{}" \\\n'.format(dst_exp_path)
            command+='--train_prior_concept1="{}" \\\n'.format(train_prior)
            command+='--eval_prior_concept1="{}" \\\n'.format(eval_prior)
            command+='--train_prompt_type="{}" \\\n'.format(train_prompt_type)
            command+='--eval_prompt_type="{}" \\\n'.format(eval_prompt_type)
            command+='--benchmark_path="../datasets_pkgs/eval_prompts/{}.json" \\\n'.format(benchmark)
            command+='--include_prior_concept=1 > {} 2>&1 &'.format(log_path)
            os.system(command)
            print('GENERATION STARTED')
            port_idx+=1
            time.sleep(30)

