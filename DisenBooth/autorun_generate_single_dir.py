from utils import float_to_str
import time
import numpy as np
import os
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
# cuda_ids=[0,1,2,3,4,5,6,7]
lambda_mlms=[
            0, 
            0.001,
            0.0025,
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

# for idx,concept in enumerate(list(info_map.keys())):
for mlm_prior_only in mlm_prior_only_list:
    mlm_prior_only_str=float_to_str(mlm_prior_only)
    mlm_prior_only_str=mlm_prior_only_str.replace('.','')
    for lambda_mlm in lambda_mlms:
        device_idx=stat_idx
        for train_text_encoder in train_text_encoders:
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
                    # run_name+='_mlmprior{}'.format(mlm_prior_only_str)
                    output_dir=os.path.join('saved_models/disenbooth_models/sd2/single',concept)
                    exp_path=os.path.join(output_dir,run_name)
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
                        print(run_name,'sleep',stat_idx,stat)
                        time.sleep(10)
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
                    command+='--max_train_steps=2001 \\\n'
                    command+='--checkpointing_steps=500 \\\n'
                    command+='--validation_steps=100 \\\n'
                    command+='--lr_warmup_steps=0 \\\n'
                    command+='--output_dir="{}" \\\n'.format(output_dir)
                    command+='--seed=7777 \\\n'
                    command+='--mask_tokens="[MASK]" \\\n'
                    command+='--lambda_mlm={} --freeze_mask_embedding=1 \\\n'.format(lambda_mlm)
                    command+='--cls_net_path="saved_models/mlm_models/sd2_contextnet_nonpadding_1e4/checkpoints/cls_net_100000_ckpt.pt" \\\n'
                    command+='--mask_embed_path="saved_models/mlm_models/sd2_contextnet_nonpadding_1e4/checkpoints/mask_embeds_100000_ckpt.pt" \\\n'
                    command+='--mlm_target=masked \\\n'
                    command+='--mlm_batch_size=50 \\\n'
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
                    time.sleep(20)
                


