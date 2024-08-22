import time
import numpy as np
import os
concepts=os.listdir('/data/twkim/diffusion/personalization/collected/images')
info_map={
    'backpack':('backpack','nonliving'),
    'teddybear':('bear','nonliving'),
    'wooden_pot':('pot','nonliving'),
    'vase':('vase','nonliving'),
    'cat1': ('cat','pet'),
    'pet_cat1':('cat','pet'),
    'pet_dog1':('dog','pet'),
    'barn': ('barn','building'),
    'chair1': ('chair','nonliving'),
    'cat_statue': ('toy','nonliving'),
    'rc_car':('toy','nonliving'),
    'pink_sunglasses':('sunglasses','sunglasses'),
    'dog3': ('dog','pet'),
    'dog6': ('dog','pet'),
    'flower1':('flower','flower'),

}
lambda_mlm=0.001


import subprocess as sp
def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values




ports=np.arange(5000,6000)
target_devices=[0,1,2,3,4,5,6,7]
stats=get_gpu_memory()
for stat_idx,stat in enumerate(stats):
    if stat>2e4 and stat_idx in target_devices:
        break
device_idx=stat_idx
idx=0
# dirs=['multi','single']
seed=2940
dir_name='singlev2_noprior_seed{}'.format(seed)
dir_path=os.path.join('saved_models/ti_models',dir_name)
log_dir='logs/generate/{}'.format(dir_name)
os.makedirs(log_dir,exist_ok=True)    
delay=30
num_images_per_prompt=8
concepts=os.listdir(dir_path)
for cidx,concept in enumerate(concepts):
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
        include_prior_concept=1
        exp_name=exp
        exp_name+='_s3000'
        output_dir=os.path.join('results/{}/{}'.format(dir_name,concept))
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
            print(exp,'sleep','{}/{}'.format(cidx+1,len(concepts)))
            time.sleep(delay)
        print(exp_name,device_idx)
        log_path=os.path.join(log_dir,exp_name+'.out')
        command='export CUDA_VISIBLE_DEVICES={};'.format(device_idx)
        command+='accelerate launch --main_process_port {} generate_single.py \\\n'.format(ports[idx],idx)
        command+='--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \\\n'
        command+='--train_data_dir1="/data/twkim/diffusion/personalization/collected/images/{}" \\\n'.format(concept)
        command+='--placeholder_token1="<{}>" \\\n'.format(concept)
        command+='--prior_concept1="{}" \\\n'.format(prior)
        command+='--resolution=512 \\\n'
        command+='--dst_exp_path={} \\\n'.format(exp_path)
        command+='--eval_batch_size=15 \\\n'
        command+='--num_images_per_prompt={} \\\n'.format(num_images_per_prompt)
        command+='--output_dir="{}" \\\n'.format(output_dir)
        command+='--seed={} \\\n'.format(seed)
        command+='--mask_tokens="[MASK]" \\\n'
        command+='--learned_embed_path1="{}" \\\n'.format(learned_embed_path1)
        command+='--prompt_type="{}" \\\n'.format(category)
        command+='--include_prior_concept={} > {} 2>&1 &'.format(include_prior_concept,log_path)
        os.system(command)
        print('STARTED')
        idx+=1
        time.sleep(15)




