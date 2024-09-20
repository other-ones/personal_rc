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
    'pet_dog1':('dog','dog','pet','living'),
    'teapot':('teapot','teapot','nonliving','nonliving'),
    'chair1': ('chair','chair','nonliving','nonliving'),
    'dog6': ('dog','dog','pet','living'),
    'duck_toy':('duck','duck toy','nonliving','nonliving'),

    'dog3':  ('dog','dog','pet','living'),
    'cat2':('cat','cat','pet','living'),

    'wooden_pot':('pot','wooden pot','nonliving','nonliving'),
    'poop_emoji':('toy','toy','nonliving','nonliving'),
    'teddybear':('teddy','teddy bear','nonliving','nonliving'),

    
    
    # 'red_cartoon':('character','cartoon character','pet','living'),
    # 'candle':('candle','candle','nonliving','nonliving'),
    # 'can':('can','can','nonliving','nonliving'),
    # 'pink_sunglasses':('sunglasses','sunglasses'),
    # 'barn': ('barn','barn'),
    # 'flower1':('flower','flower'),

}

if '03' in hostname:
    target_devices=[0,1,2,3,4,5,6,7]
elif '04' in hostname:
    target_devices=[0,1,2,3,4,5,6,7]
elif '07' in hostname:
    target_devices=[0,1,2]
    
    

lambda_mlm_list=[
            # 0, 
            0.001,
            # 0.01,
            # 0.0001,
            # 0.0005,
            # 0.00005,
            # 0.002,
            ]
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

ports=np.arange(1111,2222)
mask_prob_list=[0.15]
seed=2940
rep_id=2
if '04' in hostname:
    host_suffix='04'
elif '07' in hostname:
    host_suffix='07'
elif '03' in hostname:
    host_suffix='03'
else:
    assert False
resume_dir_name=f'init_seed{seed}_qlab{host_suffix}_rep{rep_id}'
result_dir_name=f'init_seed{seed}_qlab{host_suffix}_rep{rep_id}_valid'
# for port_idx,concept in enumerate(list(info_map.keys())):
lr_list=[1e-4]
mlm_batch_size=25
train_target_step=250
# check_tags=['VERB-ADJ-ADV-PROPN-ADP-NOUN']
check_tags=['']

print('\nTRAINING TI')
port_idx=0
train_batch_size=1





print('\n\n')
print('GENERATION')
# GENERATION
resume_dir_path=os.path.join('saved_models/cd_models',resume_dir_name)
result_dir_path=os.path.join('saved_models/cd_models',result_dir_name)
delay=30
num_images_per_prompt=8
port_idx=0
include_prior_concept=1
ppos_list=[0]
benchmark='dreambooth'
concepts=list(info_map.keys())
concepts=sorted(concepts)
gen_cd_target_step=500
gen_emb_target_step_list=[1000,2000]
for gen_emb_target_step in gen_emb_target_step_list:
    for concept_idx,concept in enumerate(list(info_map.keys())):
        resume_concept_path=os.path.join(resume_dir_path,concept)
        if not os.path.exists(resume_concept_path):
            print(resume_concept_path,'not exists',concept)
            continue
        exps=os.listdir(resume_concept_path)
        for exp_idx,exp in enumerate(exps):
            if not f'_ti{gen_cd_target_step}' in exp:
                continue

            splits=exp.split('_')
            assert 'lr' in splits[-3]
            cd_exp_name=exp.split('_lr')[0]
            cd_exp_name+='_lr1e5'
            print(cd_exp_name,'cd_exp_name')
            train_prior,eval_prior,train_prompt_type,eval_prompt_type=info_map[concept]
            resume_cd_path=os.path.join(resume_concept_path,cd_exp_name,'checkpoints/checkpoint-{}/custom_diffusion.pt'.format(gen_cd_target_step,gen_cd_target_step))
            learned_embed_path1=os.path.join(resume_concept_path,exp,'checkpoints/checkpoint-{}/learned_embeds.pt'.format(gen_emb_target_step,gen_emb_target_step))
            if not os.path.exists(resume_cd_path):
                print(resume_cd_path,'UNET does not exist')
                continue
            if not os.path.exists(learned_embed_path1):
                print(learned_embed_path1,'EMB does not exist',concept)
                continue
            ti_exp_name=exp+'_s{}'.format(gen_emb_target_step)
            output_dir=os.path.join('results/cd_results/{}/{}'.format(result_dir_name,concept))
            dst_exp_path=os.path.join(output_dir,ti_exp_name)
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
                print(f'SLEEP GENERATING {result_dir_name}\t{ti_exp_name}\t{concept_idx+1}/{len(list(info_map.keys()))}')
                time.sleep(30)
                stat_idx+=1
                stat_idx=(stat_idx%len(stats))
            print('GENERATION START\t{}\tDEVICE:{}'.format(ti_exp_name,device_idx))
            os.makedirs(dst_exp_path,exist_ok=True)  
            log_path=os.path.join(dst_exp_path,'log.out')
            command='export CUDA_VISIBLE_DEVICES={};'.format(device_idx)
            command+='export CUBLAS_WORKSPACE_CONFIG=:4096:8;'
            command+='accelerate launch --main_process_port {} cd_generate_clean.py \\\n'.format(ports[port_idx],port_idx)
            command+='--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \\\n'
            command+='--train_data_dir1="/data/twkim/diffusion/personalization/collected/images/{}" \\\n'.format(concept)
            command+='--placeholder_token1="<{}>" \\\n'.format(concept)
            command+='--resolution=512 \\\n'
            command+='--eval_batch_size=15 \\\n'
            command+='--num_images_per_prompt={} \\\n'.format(num_images_per_prompt)
            command+='--output_dir="{}" \\\n'.format(output_dir)
            command+='--seed={} \\\n'.format(seed)
            command+='--mask_tokens="[MASK]" \\\n'
            command+='--resume_cd_path="{}" \\\n'.format(resume_cd_path)
            command+='--learned_embed_path1="{}" \\\n'.format(learned_embed_path1)
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

