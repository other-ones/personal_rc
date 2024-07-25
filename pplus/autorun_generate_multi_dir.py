import time
import numpy as np
import os
concepts=os.listdir('/data/twkim/diffusion/personalization/collected/images')
info_map={
    'pet_cat1_pet_dog1':{
        'concept1':'pet_cat1',
        'concept2':'pet_dog1',
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




ports=np.arange(5000,6000)
stats=get_gpu_memory()
for stat_idx,stat in enumerate(stats):
    if stat>2e4:
        break
device_idx=stat_idx
idx=0
# dirs=['multi','single']
dirs=['multi_train']
for dir in dirs:
    dir_path=os.path.join('saved_models',dir)
    log_dir='logs/generate/{}'.format(dir)
    os.makedirs(log_dir,exist_ok=True)    
    concepts=os.listdir(dir_path)
    for joint_concept in concepts:
        if joint_concept not in info_map:
            continue
        concept_path=os.path.join(dir_path,joint_concept)
        exps=os.listdir(concept_path)
        meta_data=info_map[joint_concept]
        concept1=meta_data['concept1']
        concept2=meta_data['concept2']
        prompt_type=meta_data['prompt_type']
        prior_concept1=meta_data['prior_concept1']
        prior_concept2=meta_data['prior_concept2']
        for exp_idx,exp in enumerate(exps):
            if 'tmp' in exp:
                continue
            learned_embed_path_multi=os.path.join(concept_path,exp,'checkpoints/learned_embeds_multi_s3000.pt')
            if not os.path.exists(learned_embed_path_multi):
                print(learned_embed_path_multi,'does not exist')
                continue
            exp_name=learned_embed_path_multi.split('/')[-3]
            include_prior_concept=0
            output_dir=os.path.join('results/{}/{}'.format(dir,joint_concept))
            exp_path=os.path.join(output_dir,exp_name)
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
                time.sleep(10)
                stat_idx+=1
                stat_idx=(stat_idx%len(stats))
            print(exp_name,device_idx)
            log_path=os.path.join(log_dir,exp_name+'.out')
            command='export CUDA_VISIBLE_DEVICES={};'.format(device_idx)
            command+='accelerate launch --main_process_port {} generate_multi.py \\\n'.format(ports[idx],idx)
            command+='--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \\\n'
            command+='--train_data_dir1="/data/twkim/diffusion/personalization/collected/images/{}" \\\n'.format(concept1)
            command+='--train_data_dir2="/data/twkim/diffusion/personalization/collected/images/{}" \\\n'.format(concept2)
            command+='--placeholder_token1="<{}>" \\\n'.format(concept1)
            command+='--placeholder_token2="<{}>" \\\n'.format(concept2)
            command+='--prior_concept1="{}" \\\n'.format(prior_concept1)
            command+='--prior_concept2="{}" \\\n'.format(prior_concept2)
            command+='--resolution=512 \\\n'
            command+='--eval_batch_size=18 \\\n'
            command+='--num_images_per_prompt=15 \\\n'
            command+='--output_dir="{}" \\\n'.format(output_dir)
            command+='--seed=1234 \\\n'
            command+='--mask_tokens="[MASK]" \\\n'
            command+='--learned_embed_path_multi="{}" \\\n'.format(learned_embed_path_multi)
            command+='--prompt_type="{}" \\\n'.format(prompt_type)
            command+='--include_prior_concept={} > {} 2>&1 &'.format(include_prior_concept,log_path)
            os.system(command)
            print('STARTED')
            idx+=1
            time.sleep(15)

    


