from utils import float_to_str
import time
import numpy as np
import os
import socket
hostname = socket.gethostname()
concepts=os.listdir('/data/twkim/diffusion/personalization/collected/images')
info_map={
    'dog6': ('dog','pet'),
    'pet_cat1':('cat','pet'),
    'vase':('vase','nonliving'),
    'wooden_pot':('pot','nonliving'),
    'pet_dog1':('dog','pet'),
    'backpack':('backpack','nonliving'),
    'pink_sunglasses':('sunglasses','sunglasses'),
    'barn': ('barn','building'),
    'teddybear':('teddybear','nonliving'),
    'cat1': ('cat','pet'),
    'dog3': ('dog','pet'),
    'chair1': ('chair','nonliving'),
    'cat_statue': ('toy','nonliving'),
    'rc_car':('toy','nonliving'),
    # 'flower1':('flower','flower'),
}
lambda_mlms=[
            0.1,
            # 1.0,
            # 0.01,
            # 0.05,
            # 0.001,
            # 0, 
            # 1
            ]
masked_loss=0
if '03' in hostname:
    delay=45
    target_devices=[0,1,2,3,4,5,6,7]
else:
    delay=60
    target_devices=[0,1]

import subprocess as sp
def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values




ports=np.arange(1111,9999)
np.random.shuffle(ports)
fixtes=[0]
pps=[1]
mlm_priors=[0]

# for idx,concept in enumerate(list(info_map.keys())):
with_ti_list=[0]
noaug=0
idx=0
train_te=0
lr_list=[1e-5]
num_devices=1
for lr in lr_list:
    for lambda_mlm in lambda_mlms:
        for concept in list(info_map.keys()):
            lambda_mlm_str=float_to_str(lambda_mlm)
            lambda_mlm_str=lambda_mlm_str.replace('.','')
            prior,category=info_map[concept]
            run_name='custom'
            if lambda_mlm:
                run_name+="_mlm{}_{}".format(lambda_mlm_str,concept)
            else:
                run_name+="_nomlm_{}".format(concept)
            if train_te:
                run_name+='_train_te'
            # if lr==1e-4:
            #     run_name+='_lr1e4'
            # elif lr==1e-5:
            #     run_name+='_lr1e5'
            # else:
            #     assert False
            run_name+='_batch1_lr1e5'
            run_name+='_resume'
            # if noaug:
            #     log_dir='logs/train/single_noaug'
            #     os.makedirs(log_dir,exist_ok=True)   
            #     output_dir=os.path.join('saved_models/custom_diffusion/single_noaug',concept)
            # else:
            log_dir='logs/train/single_resume'
            os.makedirs(log_dir,exist_ok=True)   
            output_dir=os.path.join('saved_models/custom_diffusion/single_resume',concept)
            exp_path=os.path.join(output_dir,run_name)
            if os.path.exists(exp_path):
                print(exp_path,'exists')
                continue
            while True:
                idle_devices=[]
                stats=get_gpu_memory()
                for device_idx in target_devices:
                    stat=stats[device_idx]   
                    if stat>2e4:
                        idle_devices.append(str(device_idx))
                    else:
                        print(device_idx,'not available')
                    idx+=1
                if len(idle_devices)>=num_devices:
                    idx+=1
                    break
                print(run_name,'sleep')
                time.sleep(delay)
            log_path=os.path.join(log_dir,run_name+'.out')
            print(log_path,'log_path')
            running_devices=','.join(idle_devices[:num_devices])
            resume_path=os.path.join("saved_models/custom_diffusion/single/{}/custom_nomlm_{}/checkpoints/checkpoint-250/".format(concept,concept,concept))
            # print(resume_path,'resume')
            # assert os.path.exists(resume_path)
            if not os.path.exists(resume_path):
                continue
            print(run_name,running_devices)
            command='export CUDA_VISIBLE_DEVICES={};'.format(running_devices)
            command+='accelerate launch --main_process_port {} train_custom_diffusion_single.py \\\n'.format(ports[idx],idx)
            command+='--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \\\n'
            command+='--train_data_dir1="/data/twkim/diffusion/personalization/collected/images/{}" \\\n'.format(concept)
            command+='--placeholder_token1="<{}>" \\\n'.format(concept)
            command+='--prior_concept1="{}" \\\n'.format(prior)
            command+='--resolution=512 \\\n'
            command+='--resume_path=\"{}\" \\\n'.format(resume_path)
            command+='--train_batch_size=1 \\\n'
            command+='--gradient_accumulation_steps=1 \\\n'
            command+='--max_train_steps=3001 \\\n'
            command+='--validation_steps=100 \\\n'
            # command+='--checkpoints_total_limit=1 \\\n'
            command+='--checkpointing_steps=500 \\\n'
            command+='--learning_rate={} \\\n'.format(lr)
            command+='--lr_scheduler="constant" \\\n'
            command+='--lr_warmup_steps=0 \\\n'
            command+='--output_dir="{}" \\\n'.format(output_dir)
            command+='--train_text_encoder={} \\\n'.format(train_te)
            command+='--seed=7777 \\\n'
            command+='--mask_tokens="[MASK]" \\\n'
            command+='--lambda_mlm={} --freeze_mask_embedding=1 \\\n'.format(lambda_mlm)
            command+='--cls_net_path="saved_models/mlm_models/mlm_contextnet_nonpad_lr1e4/checkpoints/cls_net_99000_ckpt.pt" \\\n'
            command+='--mask_embed_path="saved_models/mlm_models/mlm_contextnet_nonpad_lr1e4/checkpoints/mask_embeds_99000_ckpt.pt" \\\n'
            command+='--mlm_target=masked \\\n'
            command+='--mlm_batch_size=50 \\\n'
            command+='--prompt_type="{}" \\\n'.format(category)
            command+='--scale_lr \\\n'
            command+='--noaug={} \\\n'.format(noaug)
            command+='--silent=0 \\\n'
            command+='--simple_caption=1 \\\n'
            command+='--masked_loss={} \\\n'.format(masked_loss)
            command+='--normalize_target1=0 \\\n'
            command+='--run_name="{}" \\\n'.format(run_name)
            command+='--with_prior_preservation=0 \\\n'
            command+='--class_prompt1="a picture of a {}" \\\n'.format(prior)
            command+='--class_data_dir1="priors/samples_{}" \\\n'.format(prior)
            # command+='--report_to="wandb" \\\n'
            # command+='--project_name="DreamBooth MLM SINGLE" \\\n'
            command+='--include_prior_concept=1 > {} 2>&1 &'.format(log_path)
            os.system(command)
            print('STARTED')
            # exit()
            idx+=1
            time.sleep(delay)
            


