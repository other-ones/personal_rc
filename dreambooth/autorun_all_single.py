from utils import float_to_str,invert_scientific_notation
import time
import numpy as np
import os
import socket
hostname = socket.gethostname()
print(hostname,'hostname')
concepts=os.listdir('/data/twkim/diffusion/personalization/collected/images')
info_map_03={
    # train_prior/eval_prior/train_prompt_type/eval_prompt_type
    'dog6': ('dog','dog','pet','living'),
    'duck_toy':('duck','duck toy','nonliving','nonliving'),
    'pet_cat1':('cat','cat','pet','living'),
    # 'teapot':('teapot','teapot','nonliving','nonliving'),

    'wooden_pot':('pot','wooden pot','nonliving','nonliving'),
    # 'backpack_dog':('backpack','backpack','nonliving','nonliving'),
    # 'poop_emoji':('toy','toy','nonliving','nonliving'),
    # 'cat2':('cat','cat','pet','living'),
    # 'cat1': ('cat','cat','pet','living'),
    # 'dog3':  ('dog','dog','pet','living'),
    # 'pet_dog1':('dog','dog','pet','living'),
    # 'backpack':('backpack','backpack','nonliving','nonliving'),
    # 'teddybear':('bear','teddy bear','nonliving','nonliving'),
    # 'cat_statue': ('toy','toy','nonliving','nonliving'),
    # 'rc_car':('toy','toy','nonliving','nonliving'),
    # 'chair1': ('chair','chair','nonliving','nonliving'),

    # 'red_cartoon':('character','cartoon character','pet','living'),
    # 'candle':('candle','candle','nonliving','nonliving'),
    # 'can':('can','can','nonliving','nonliving'),
    # 'pink_sunglasses':('sunglasses','sunglasses'),
    # 'barn': ('barn','barn'),
    # 'flower1':('flower','flower'),
}
info_map_01={
    # train_prior/eval_prior/train_prompt_type/eval_prompt_type
    'teapot':('teapot','teapot','nonliving','nonliving'),
    'dog6': ('dog','dog','pet','living'),
    'duck_toy':('duck','duck toy','nonliving','nonliving'),
    'pet_cat1':('cat','cat','pet','living'),

    'wooden_pot':('pot','wooden pot','nonliving','nonliving'),
    'backpack_dog':('backpack','backpack','nonliving','nonliving'),
    'poop_emoji':('toy','toy','nonliving','nonliving'),
    'cat2':('cat','cat','pet','living'),
    'cat1': ('cat','cat','pet','living'),
    'dog3':  ('dog','dog','pet','living'),
    'pet_dog1':('dog','dog','pet','living'),
    'backpack':('backpack','backpack','nonliving','nonliving'),
    'teddybear':('bear','teddy bear','nonliving','nonliving'),
    'cat_statue': ('toy','toy','nonliving','nonliving'),
    'rc_car':('toy','toy','nonliving','nonliving'),
    'chair1': ('chair','chair','nonliving','nonliving'),

    # 'red_cartoon':('character','cartoon character','pet','living'),
    # 'candle':('candle','candle','nonliving','nonliving'),
    # 'can':('can','can','nonliving','nonliving'),
    # 'pink_sunglasses':('sunglasses','sunglasses'),
    # 'barn': ('barn','barn'),
    # 'flower1':('flower','flower'),
}
if '03' in hostname:
    info_map=info_map_03
elif 'ubuntu' in hostname:
    info_map=info_map_01

target_devices=[0,1,2,3,4,5,6,7]
lambda_mlm_list=[
            0, 
            0.001,
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
fixte_list=[0]
mask_prob_list=[0.25]
seed=2940
dir_name='single_capv7_seed{}'.format(seed)
log_dir='logs/train/{}'.format(dir_name)
os.makedirs(log_dir,exist_ok=True)   
# for port_idx,concept in enumerate(list(info_map.keys())):
lr_list=[1e-6]
for lr in lr_list:
    lr_str=invert_scientific_notation(lr)
    lr_str=lr_str.replace('.','P')
    for mask_prob in mask_prob_list:
        mask_prob_str=float_to_str(mask_prob)
        mask_prob_str=mask_prob_str.replace('.','')
        for port_idx,concept in enumerate(list(info_map.keys())):
            device_idx=stat_idx
            for fixte in fixte_list:
                for lambda_mlm in lambda_mlm_list:
                    lambda_mlm_str=float_to_str(lambda_mlm)
                    lambda_mlm_str=lambda_mlm_str.replace('.','')
                    train_prior,eval_prior,train_prompt_type,eval_prompt_type=info_map[concept]
                    run_name='db_cnetv4'
                    if lambda_mlm:
                        run_name+="_mlm{}_{}".format(lambda_mlm_str,concept)
                    else:
                        run_name+="_nomlm_{}".format(concept)
                    if fixte:
                        run_name+='_fixte'
                    if lambda_mlm:
                        run_name+='_mprob{}'.format(mask_prob_str)
                    run_name+='_lr{}'.format(lr_str)
                    output_dir=os.path.join('saved_models/db_models/{}'.format(dir_name),concept)
                    exp_path=os.path.join(output_dir,run_name)
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
                        print(run_name,'sleep',stat_idx,stat)
                        time.sleep(10)
                    print(run_name,device_idx)
                    log_path=os.path.join(log_dir,run_name+'.out')
                    command='export CUDA_VISIBLE_DEVICES={};'.format(device_idx)
                    command+='export CUBLAS_WORKSPACE_CONFIG=:4096:8;'
                    command+='accelerate launch --main_process_port {} db_train.py \\\n'.format(ports[port_idx])
                    command+='--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \\\n'
                    command+='--train_data_dir1="/data/twkim/diffusion/personalization/collected/images/{}" \\\n'.format(concept)
                    command+='--placeholder_token1="<{}>" \\\n'.format(concept)
                    command+='--train_prior_concept1="{}" \\\n'.format(train_prior)
                    command+='--eval_prior_concept1="{}" \\\n'.format(eval_prior)
                    command+='--eval_prompt_type="{}" \\\n'.format(eval_prompt_type)
                    command+='--train_prompt_type="{}" \\\n'.format(train_prompt_type)
                    command+='--resolution=512 \\\n'
                    command+='--train_batch_size=1 \\\n'
                    command+='--gradient_accumulation_steps=1 \\\n'
                    command+='--max_train_steps=1001 \\\n'
                    command+='--learning_rate={} \\\n'.format(lr)
                    command+='--lr_scheduler="constant" \\\n'
                    command+='--lr_warmup_steps=0 \\\n'
                    command+='--output_dir="{}" \\\n'.format(output_dir)
                    command+='--seed={} \\\n'.format(seed)
                    command+='--mask_tokens="[MASK]" \\\n'
                    command+='--lambda_mlm={} --freeze_mask_embedding=1 \\\n'.format(lambda_mlm)
                    command+='--cls_net_path="saved_models/mlm_models/sd1_contextnetv4_nonpadding_1e4_unnorm_mprob015_batch150/checkpoints/checkpoint-100000/cls_net_100000_ckpt.pt" \\\n'
                    command+='--mask_embed_path="saved_models/mlm_models/sd1_contextnetv4_nonpadding_1e4_unnorm_mprob015_batch150/checkpoints/checkpoint-100000/mask_embeds_100000_ckpt.pt" \\\n'
                    # command+='--cls_net_path="saved_models/mlm_models/sd1_contextnetv3_nonpadding_1e4/checkpoints/checkpoint-100000/cls_net_100000_ckpt.pt" \\\n'
                    # command+='--mask_embed_path="saved_models/mlm_models/sd1_contextnetv3_nonpadding_1e4/checkpoints/checkpoint-100000/mask_embeds_100000_ckpt.pt" \\\n'
                    command+='--mlm_target=masked \\\n'
                    command+='--mlm_batch_size=20 \\\n'
                    command+='--mask_prob={} \\\n'.format(mask_prob)
                    command+='--silent=0 \\\n'
                    command+='--simple_caption=0 \\\n'
                    command+='--masked_loss={} \\\n'.format(masked_loss)
                    command+='--normalize_target1=0 \\\n'
                    command+='--run_name="{}" \\\n'.format(run_name)
                    command+='--with_prior_preservation={} \\\n'.format(1)
                    command+='--class_prompt1="a picture of a {}" \\\n'.format(train_prior)
                    command+='--class_data_dir1="priors/samples_{}" \\\n'.format(train_prior)
                    command+='--caption_root="../datasets_pkgs/captions/v7" \\\n'
                    if fixte==0: # do not fix text_encoder
                        command+='--train_text_encoder \\\n'
                    # command+='--report_to="wandb" \\\n'
                    # command+='--project_name="DreamBooth MLM SINGLE" \\\n'
                    command+='--include_prior_concept=1 > {} 2>&1 &'.format(log_path)
                    os.system(command)
                    print('STARTED')
                    time.sleep(15)
                


