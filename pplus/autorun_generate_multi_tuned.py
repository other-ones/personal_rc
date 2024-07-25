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
lambda_mlm=0.001


import subprocess as sp
def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values
def float_to_str(f):
    s = f"{f:.15f}"  # Start with a high precision
    return s.rstrip('0').rstrip('.') if '.' in s else s

log_dir='logs/generate/multi'
os.makedirs(log_dir,exist_ok=True)    
ports=np.arange(4000,5000)
stats=get_gpu_memory()
for stat_idx,stat in enumerate(stats):
    if stat>2e4:
        break
device_idx=stat_idx
target_norms=[0]
concept_pairs=list(info_map.keys())
lambda_contrastives=[0]
lambda_mlm_str=float_to_str(lambda_mlm)
lambda_mlm_str=lambda_mlm_str.replace('.','')
sim_margins=[0.2,0.3,0.4,0.1,0.5]
lambda_sims=[0.01,0.001]
for lambda_sim in lambda_sims:
    lambda_sim_str=float_to_str(lambda_sim)
    lambda_sim_str=lambda_sim_str.replace('.','')
    for sim_margin in sim_margins:
        sim_margin_str=float_to_str(sim_margin)
        sim_margin_str=sim_margin_str.replace('.','')
        for lambda_contrastive in lambda_contrastives:
            lambda_contrastive_str=float_to_str(lambda_contrastive)
            lambda_contrastive_str=lambda_contrastive_str.replace('.','')
            for target_norm in target_norms:
                for idx,concept_pair in enumerate(concept_pairs):
                    meta_data=info_map[concept_pair]
                    concept1=concept_pair[0]
                    concept2=concept_pair[1]
                    joint_concept='{}_{}'.format(concept1,concept2)
                    prior_concept1=meta_data['prior_concept1']
                    prior_concept2=meta_data['prior_concept2']
                    prompt_type=meta_data['prompt_type']
                    saved_dir='saved_models/multi/{}'.format(joint_concept)
                    # ti_multi_margin05_sim0001_mlm0001_contrast0_pet_cat1_pet_dog1
                    learned_embed_path_multi='{}/ti_multi_margin{}_sim{}_mlm{}_contrast{}_{}'.format(saved_dir,sim_margin_str,lambda_sim_str,lambda_mlm_str,lambda_contrastive_str,joint_concept)
                    learned_embed_path_multi+='/checkpoints/learned_embeds_multi_s3000.pt'
                    if not os.path.exists(learned_embed_path_multi):
                        print(learned_embed_path_multi,'does not exists',target_norm)
                        continue
                    exp_name=learned_embed_path_multi.split('/')[-3]
                    output_dir=os.path.join('results/multi/{}'.format(joint_concept))
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
                        print('sleep waiting for {}'.format(exp_name),'GPU[{}] is busy FREE: {}MB'.format(stat_idx,stat),'# Remaining Exps: {}'.format(len(info_map)-idx))
                        time.sleep(10)
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
                    command+='--learned_embed_path_multi="{}" \\\n'.format(learned_embed_path_multi)
                    command+='--prior_concept1="{}" \\\n'.format(prior_concept1)
                    command+='--prior_concept2="{}" \\\n'.format(prior_concept2)
                    command+='--resolution=512 \\\n'
                    command+='--eval_batch_size=18 \\\n'
                    command+='--num_images_per_prompt=15 \\\n'
                    command+='--output_dir="{}" \\\n'.format(output_dir)
                    command+='--seed=1234 \\\n'
                    command+='--mask_tokens="[MASK]" \\\n'
                    command+='--prompt_type="{}" \\\n'.format(prompt_type)
                    command+='--normalize_target1={} \\\n'.format(target_norm)
                    command+='--normalize_target2={} \\\n'.format(target_norm)
                    command+='--include_prior_concept=0 > {} 2>&1 &'.format(log_path)
                    os.system(command)
                    print('STARTED')
                    time.sleep(15)

                


