from utils import float_to_str
import time
import numpy as np
import os
import socket
hostname = socket.gethostname()
print(hostname,'hostname')
concepts=os.listdir('/data/twkim/diffusion/personalization/collected/images')
info_map={
    'dog6': ('dog','pet'),
    'pet_cat1':('cat','pet'),
    'vase':('vase','nonliving'),
    'wooden_pot':('pot','nonliving'),
    'backpack':('backpack','nonliving'),
    'teddybear':('teddybear','nonliving'),
    'cat1': ('cat','pet'),
    'pet_dog1':('dog','pet'),
    'barn': ('barn','building'),
    'chair1': ('chair','nonliving'),
    'cat_statue': ('toy','nonliving'),
    'rc_car':('toy','nonliving'),
    # 'pink_sunglasses':('sunglasses','sunglasses'),
    'dog3': ('dog','pet'),
    # 'flower1':('flower','flower'),

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
if '03' in hostname:
    delay=45
    target_devices=[0,1,2,3,4,5,6,7]
else:
    delay=45
    target_devices=[0,1]
dirs=['single']
for dir in dirs:
    dir_path=os.path.join('saved_models/dreambooth_models',dir)
    log_dir='logs/generate/{}'.format(dir)
    os.makedirs(log_dir,exist_ok=True)    
    concepts=os.listdir(dir_path)
    concepts=sorted(concepts)
    for concept in concepts:
        if concept not in info_map:
            continue
        concept_path=os.path.join(dir_path,concept)
        exps=os.listdir(concept_path)
        for exp_idx,exp in enumerate(exps):
            prior,category=info_map[concept]
            resume_unet_path=os.path.join(concept_path,exp,'checkpoints/checkpoint-1000/unet_s1000.pt')
            resume_text_encoder_path=os.path.join(concept_path,exp,'checkpoints/checkpoint-1000/text_encoder_s1000.pt')
            if not os.path.exists(resume_unet_path):
                print(resume_unet_path,'does not exist')
                continue
            exp_name=resume_unet_path.split('/')[-4]
            output_dir=os.path.join('results/{}/{}'.format(dir,concept))
            exp_path=os.path.join(output_dir,exp_name)
            if os.path.exists(exp_path):
                print(exp_name,'exists')
                continue
            while True:
                stats=get_gpu_memory()
                stat=stats[stat_idx%len(stats)]
                found=False
                for stat_idx in target_devices:
                    stat=stats[stat_idx]    
                    if stat>2e4:
                        device_idx=stat_idx
                        stat_idx+=1
                        found=True
                        break
                    time.sleep(5)
                if found:
                    break
                print('sleep waiting for {}'.format(exp_name))
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
            command+='--mask_tokens="[MASK]" \\\n'
            command+='--resume_unet_path="{}" \\\n'.format(resume_unet_path)
            command+='--resume_text_encoder_path="{}" \\\n'.format(resume_text_encoder_path)
            command+='--prompt_type="{}" \\\n'.format(category)
            command+='--include_prior_concept=1 > {} 2>&1 &'.format(log_path)
            os.system(command)
            print('STARTED')
            idx+=1
            time.sleep(15)

    


