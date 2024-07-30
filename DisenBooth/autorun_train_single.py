from utils import format_exponent
from utils import float_to_str
import time
import numpy as np
import os
import socket
hostname = socket.gethostname()
print(hostname,'hostname')
info_map_03={
    'pet_dog1':('dog','pet'),
    'backpack':('backpack','nonliving'),
    'pet_cat1':('cat','pet'),
    'vase':('vase','nonliving'),
    'cat1': ('cat','pet'),
    'barn': ('barn','building'),
    'wooden_pot':('pot','nonliving'),
    # 'dog3': ('dog','pet'),
    # 'chair1': ('chair','nonliving'),
    # 'cat_statue': ('toy','nonliving'),
    # 'rc_car':('toy','nonliving'),
    # 'pink_sunglasses':('sunglasses','sunglasses'),
    # 'teddybear':('bear','nonliving'),
    # 'dog6': ('dog','pet'),
}
info_map_01={
    # 'pet_dog1':('dog','pet'),
    # 'backpack':('backpack','nonliving'),
    # 'pet_cat1':('cat','pet'),
    # 'vase':('vase','nonliving'),
    # 'cat1': ('cat','pet'),
    # 'barn': ('barn','building'),
    # 'wooden_pot':('pot','nonliving'),

    'dog3': ('dog','pet'),
    'chair1': ('chair','nonliving'),
    'cat_statue': ('toy','nonliving'),
    'rc_car':('toy','nonliving'),
    'pink_sunglasses':('sunglasses','sunglasses'),
    'teddybear':('bear','nonliving'),
    'dog6': ('dog','pet'),
}
if '03' in hostname:
    info_map=info_map_03
elif 'ubuntu' in hostname:
    info_map=info_map_01
# cuda_ids=[0,1,2,3,4,5,6,7]
concepts=os.listdir('/data/twkim/diffusion/personalization/collected/images')

lambda_mlms=[
            0, 
            0.001,
            0.01,
            # 0.1,
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
log_dir='logs/disenbooth/train/sd2/single/'
os.makedirs(log_dir,exist_ok=True)   
ports=np.arange(1111,2222)
train_text_encoders=[1]
pps=[0]

mlm_prior_only_list=[0]
learning_rate=1e-5
learning_rate_adapter=1e-4
learning_rate_str=format_exponent(learning_rate)
learning_rate_adapter_str=format_exponent(learning_rate_adapter)
# print(learning_rate_str,'learning_rate_str')
# print(adapter_learning_rate_str,'adapter_learning_rate_str')
# exit()
# for idx,concept in enumerate(list(info_map.keys())):
if '03' in hostname:
    delay=30
    target_devices=[0,1,2,3,4,5,6,7]
else:
    delay=45
    target_devices=[0,1]
for mlm_prior_only in mlm_prior_only_list:
    mlm_prior_only_str=float_to_str(mlm_prior_only)
    mlm_prior_only_str=mlm_prior_only_str.replace('.','')
    for lambda_mlm in lambda_mlms:
        device_idx=stat_idx
        if not lambda_mlm: #lambda_mlm==0 -> do not train text_encoder
            train_text_encoder=0
        else:
            train_text_encoder=1
        for pp in pps:
            # for lambda_mlm in lambda_mlms:
            for idx,concept in enumerate(list(info_map.keys())):
                lambda_mlm_str=float_to_str(lambda_mlm)
                lambda_mlm_str=lambda_mlm_str.replace('.','')
                prior,category=info_map[concept]
                prefix='disenbooth'
                if pp:
                    run_name="{}_pp".format(prefix)
                else:
                    run_name="{}_nopp".format(prefix)
                # run_name=prefix
                if lambda_mlm:
                    run_name+="_mlm{}_{}".format(lambda_mlm_str,concept)
                else:
                    run_name+="_nomlm_{}".format(concept)
                if masked_loss:
                    run_name+='_masked'
                if train_text_encoder:
                    run_name+='_train_text'
                run_name+='_lr{}_alr{}'.format(learning_rate_str,learning_rate_adapter_str)
                # run_name+='_mlmprior{}'.format(mlm_prior_only_str)
                output_dir=os.path.join('saved_models/disenbooth_models/sd2/single',concept)
                exp_path=os.path.join(output_dir,run_name)
                if os.path.exists(exp_path):
                    print(exp_path,'exists')
                    continue
                while True:
                    stats=get_gpu_memory()
                    found=False
                    for stat_idx in target_devices:
                        stat=stats[stat_idx]    
                        if stat>2e4:
                            device_idx=stat_idx
                            stat_idx+=1
                            found=True
                            break
                    if found:
                        break
                    print(run_name,'sleep',stat_idx,stat)
                    time.sleep(delay)
                    stat_idx+=1
                    stat_idx=(stat_idx%len(stats))
                print(run_name,device_idx)
                log_path=os.path.join(log_dir,run_name+'.out')
                command='export CUDA_VISIBLE_DEVICES={};'.format(device_idx)
                command+='accelerate launch --main_process_port {} train_disenbooth_single.py \\\n'.format(ports[idx],idx)
                command+='--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \\\n'
                command+='--train_data_dir1="/data/twkim/diffusion/personalization/collected/images/{}" \\\n'.format(concept)
                command+='--placeholder_token1="<{}>" \\\n'.format(concept)
                command+='--prior_concept1="{}" \\\n'.format(prior)
                command+='--resolution=512 \\\n'
                command+='--train_batch_size=1 \\\n'
                command+='--gradient_accumulation_steps=1 \\\n'
                command+='--max_train_steps=3001 \\\n'
                command+='--checkpointing_steps=500 \\\n'
                command+='--validation_steps=100 \\\n'
                command+='--learning_rate={} \\\n'.format(learning_rate)
                command+='--learning_rate_adapter={} \\\n'.format(learning_rate_adapter)
                command+='--lr_scheduler="constant" \\\n'
                command+='--lr_warmup_steps=0 \\\n'
                command+='--output_dir="{}" \\\n'.format(output_dir)
                command+='--seed=7777 \\\n'
                command+='--mask_tokens="[MASK]" \\\n'
                command+='--lambda_mlm={} --freeze_mask_embedding=1 \\\n'.format(lambda_mlm)
                command+='--cls_net_path="saved_models/mlm_models/sd2_contextnet_nonpadding_1e4/checkpoints/cls_net_100000_ckpt.pt" \\\n'
                command+='--mask_embed_path="saved_models/mlm_models/sd2_contextnet_nonpadding_1e4/checkpoints/mask_embeds_100000_ckpt.pt" \\\n'
                command+='--mlm_target=masked \\\n'
                command+='--mlm_batch_size=30 \\\n'
                command+='--prompt_type="{}" \\\n'.format(category)
                command+='--silent=0 \\\n'
                command+='--simple_caption=1 \\\n'
                command+='--masked_loss={} \\\n'.format(masked_loss)
                command+='--normalize_target1=0 \\\n'
                command+='--mlm_prior_only={} \\\n'.format(mlm_prior_only)
                command+='--run_name="{}" \\\n'.format(run_name)
                if train_text_encoder:
                    command+='--train_text_encoder \\\n'
                # command+='--report_to="wandb" \\\n'
                # command+='--project_name="disenbooth MLM SINGLE" \\\n'
                command+='--include_prior_concept=1 > {} 2>&1 &'.format(log_path)
                os.system(command)
                print('STARTED')
                time.sleep(delay)
                


