import time
import numpy as np
import os
concepts=os.listdir('/data/twkim/diffusion/personalization/collected/images')
info_map={
    ('pet_cat1','pet_dog1'):{
        'prior_concept1':'cat',
        'prior_concept2':'dog',
        'prompt_type':'two_pets',
    },
    # 'teddybear':('teddybear','nonliving'),
    # 'wooden_pot':('pot','nonliving'),
    # 'vase':('vase','nonliving'),
    # 'cat1': ('cat','pet'),
    # 'pet_cat1':('cat','pet'),
    # 'pet_dog1':('dog','pet'),
    # 'barn': ('barn','building'),

    # 'chair1': ('chair','nonliving'),
    # 'cat_statue': ('toy','nonliving'),
    # 'rc_car':('toy','nonliving'),
    # 'pink_sunglasses':('sunglasses','sunglasses'),
    # 'flower1':('flower','flower'),
    # 'dog3': ('dog','pet'),
    # 'dog6': ('dog','pet'),
}
lambda_mlm=0


import subprocess as sp
def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values
def float_to_str(f):
    s = f"{f:.15f}"  # Start with a high precision
    return s.rstrip('0').rstrip('.') if '.' in s else s



dir_name='multi_raw'
log_dir='logs/generate/{}'.format(dir_name)
os.makedirs(log_dir,exist_ok=True)    
ports=np.arange(4000,5000)
stats=get_gpu_memory()
for stat_idx,stat in enumerate(stats):
    if stat>2e4:
        break
device_idx=stat_idx
# target_norms=[0,4,8,16]
target_norms=[0]
concept_pairs=list(info_map.keys())
lambda_mlm_str=float_to_str(lambda_mlm)
lambda_mlm_str=lambda_mlm_str.replace('.','')

for target_norm in target_norms:
    for idx,concept_pair in enumerate(concept_pairs):
        meta_data=info_map[concept_pair]
        concept1=concept_pair[0]
        concept2=concept_pair[1]
        joint_concept='{}_{}'.format(concept1,concept2)
        prior_concept1=meta_data['prior_concept1']
        prior_concept2=meta_data['prior_concept2']
        prompt_type=meta_data['prompt_type']
        saved_dir1='saved_models/single/{}'.format(concept1)
        saved_dir2='saved_models/single/{}'.format(concept2)
        if not lambda_mlm:
            learned_embed_path1='{}/ti_noprior_nomlm_{}/checkpoints/learned_embeds_s3000.pt'.format(saved_dir1,concept1)
            learned_embed_path2='{}/ti_noprior_nomlm_{}/checkpoints/learned_embeds_s3000.pt'.format(saved_dir2,concept2)
            exp_name='ti_noprior_rawmulti_nomlm_{}'.format(joint_concept)

        else:
            learned_embed_path1='{}/ti_norm{}_noprior_mlm{}_{}/checkpoints/learned_embeds_s3000.pt'.format(saved_dir1,target_norm,lambda_mlm_str,concept1)
            learned_embed_path2='{}/ti_norm{}_noprior_mlm{}_{}/checkpoints/learned_embeds_s3000.pt'.format(saved_dir2,target_norm,lambda_mlm_str,concept2)
            exp_name='ti_norm{}_noprior_rawmulti_mlm0001_{}'.format(target_norm,joint_concept)
        if not (os.path.exists(learned_embed_path1) and os.path.exists(learned_embed_path2)):
            print(learned_embed_path1,'does not exists')
            print(learned_embed_path2,'does not exists')
            continue
        output_dir=os.path.join('results/{}/{}'.format(dir_name,joint_concept))
        exp_path=os.path.join(output_dir,exp_name)
        if os.path.exists(exp_path):
            print(exp_path,'exists')
            continue

        # Find Free GPU 
        while True:
            stats=get_gpu_memory()
            stat=stats[stat_idx%len(stats)]
            if stat>2e4:
                device_idx=stat_idx
                stat_idx+=1
                break
            print('sleep waiting for {}'.format(exp_name),'GPU[{}] is busy. FREE Memory: {}MB'.format(stat_idx,stat),'# Remaining Exps: {}'.format(len(info_map)-idx))
            time.sleep(20)
            stat_idx+=1
            stat_idx=(stat_idx%len(stats))
        # Find Free GPU 
        print(exp_name,device_idx)
        log_path=os.path.join(log_dir,exp_name+'.out')
        command='export CUDA_VISIBLE_DEVICES={};'.format(device_idx)
        command+='accelerate launch --main_process_port {} generate_multi.py \\\n'.format(ports[idx],idx)
        command+='--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \\\n'
        command+='--train_data_dir1="/data/twkim/diffusion/personalization/collected/images/{}" \\\n'.format(concept1)
        command+='--train_data_dir2="/data/twkim/diffusion/personalization/collected/images/{}" \\\n'.format(concept2)
        command+='--placeholder_token1="<{}>" \\\n'.format(concept1)
        command+='--placeholder_token2="<{}>" \\\n'.format(concept2)
        command+='--learned_embed_path1="{}" \\\n'.format(learned_embed_path1)
        command+='--learned_embed_path2="{}" \\\n'.format(learned_embed_path2)
        command+='--prior_concept1="{}" \\\n'.format(prior_concept1)
        command+='--prior_concept2="{}" \\\n'.format(prior_concept2)
        command+='--resolution=512 \\\n'
        command+='--eval_batch_size=18 \\\n'
        command+='--num_images_per_prompt=15 \\\n'
        command+='--output_dir="{}" \\\n'.format(output_dir)
        command+='--exp_name="{}" \\\n'.format(exp_name)
        command+='--seed=1234 \\\n'
        command+='--mask_tokens="[MASK]" \\\n'
        command+='--prompt_type="{}" \\\n'.format(prompt_type)
        command+='--normalize_target1={} \\\n'.format(target_norm)
        command+='--normalize_target2={} \\\n'.format(target_norm)
        command+='--include_prior_concept=0 > {} 2>&1 &'.format(log_path)
        os.system(command)
        print('STARTED')
        time.sleep(15)

        


