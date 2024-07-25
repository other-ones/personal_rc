import time
import numpy as np
import os
concepts=os.listdir('/data/twkim/diffusion/personalization/collected/images')
info_map={
    'backpack':('backpack','nonliving'),
    'teddybear':('teddybear','nonliving'),
    'wooden_pot':('pot','nonliving'),
    'vase':('vase','nonliving'),
    'cat1': ('cat','pet'),
    'pet_cat1':('cat','pet'),
    'pet_dog1':('dog','pet'),
    'barn': ('barn','building'),


    # 'chair1': ('chair','nonliving'),
    # 'cat_statue': ('toy','nonliving'),
    # 'rc_car':('toy','nonliving'),
    # 'pink_sunglasses':('sunglasses','sunglasses'),
    # 'flower1':('flower','flower'),
    # 'dog3': ('dog','pet'),
    # 'dog6': ('dog','pet'),
}
# cuda_ids=[0,1,2,3,4,5,6,7]
cuda_ids=[0,1,2,3]
lambda_mlm=0.001


import subprocess as sp
def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


log_dir='logs/generate/single'
os.makedirs(log_dir,exist_ok=True)    
lambda_mlm_str=str(lambda_mlm).replace('.','')

ports=np.arange(3000,4000)
stat_idx=0
device_idx=stat_idx


for idx,concept in enumerate(list(info_map.keys())):
    prior,category=info_map[concept]
    learned_embed_path1='saved_models/single/{}/ti_noprior_nomlm_{}/checkpoints/learned_embeds_s3000.pt'.format(concept,concept)
    if not os.path.exists(learned_embed_path1):
        print(learned_embed_path1,'does not exists')
        continue
    exp_name=learned_embed_path1.split('/')[-3]
    output_dir=os.path.join('results/single/{}'.format(concept))
    exp_path=os.path.join(output_dir,exp_name)
    if os.path.exists(exp_path):
        print(exp_path,'exists')
        continue
    while True:
        stats=get_gpu_memory()
        stat=stats[stat_idx%len(stats)]
        if stat>2e4:
            device_idx=stat_idx
            stat_idx+=1
            break
        print('sleep waiting for {}'.format(exp_name),'GPU[{}] is busy FREE: {}MB'.format(stat_idx,stat),'# Remaining Exps: {}'.format(len(info_map)-idx))
        time.sleep(20)
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
    command+='--mask_tokens="[MASK]" \\\n'
    command+='--learned_embed_path1="{}" \\\n'.format(learned_embed_path1)
    command+='--prompt_type="{}" \\\n'.format(category)
    command+='--include_prior_concept=0 > {} 2>&1 &'.format(log_path)
    os.system(command)
    print('STARTED')
    time.sleep(20)

    


