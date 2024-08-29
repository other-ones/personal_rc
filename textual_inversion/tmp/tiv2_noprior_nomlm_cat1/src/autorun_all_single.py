from utils import float_to_str
import time
import numpy as np
import os
import socket
hostname = socket.gethostname()
print(hostname,'hostname')
concepts=os.listdir('/data/twkim/diffusion/personalization/collected/images')
info_map_03={
    'dog6': ('dog','dog'),
    'pet_cat1':('cat','cat'),
    # 'cat1': ('cat','cat'),
    # 'dog3': ('dog','dog'),
    # 'pet_dog1':('dog','dog'),
    # 'vase':('vase','vase'),
    # 'backpack':('backpack','backpack'),
    # 'teddybear':('bear','teddybear'),
    # 'barn': ('barn','barn'),
    # 'wooden_pot':('pot','wooden_pot'),
    # 'chair1': ('chair','chair'),
    # 'cat_statue': ('toy','toy'),
    # 'rc_car':('toy','toy'),
    # 'pink_sunglasses':('sunglasses','sunglasses'),
    # 'flower1':('flower','flower'),
}
info_map_01={
    # 'backpack':('backpack','nonliving'),
    # 'pet_cat1':('cat','pet'),
    # 'pet_dog1':('dog','pet'),
    # 'vase':('vase','nonliving'),
    'teddybear':('bear','nonliving'),
    'dog6': ('dog','pet'),
    # 'cat1': ('cat','pet'),
    # 'barn': ('barn','building'),
    # 'wooden_pot':('pot','nonliving'),

    # 'dog3': ('dog','pet'),
    # 'chair1': ('chair','nonliving'),
    # 'cat_statue': ('toy','nonliving'),
    # 'rc_car':('toy','nonliving'),
    # 'pink_sunglasses':('sunglasses','sunglasses'),
    # 'flower1':('flower','flower'),
}
if '03' in hostname:
    info_map=info_map_03
elif 'ubuntu' in hostname:
    info_map=info_map_01
# cuda_ids
# cuda_ids=[0,1,2,3,4,5,6,7]
lambda_mlm_list=[
            # 0,
            0.001,
            # 0.005,
            # 0.01,
            # 0.0005,
            # 0.0002,x
            ]
target_norms=[0]


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
np.random.shuffle(ports)
target_devices=[0,1,2,3,4,5,6,7]
seed=2940
include_prior=1
delay=25
mask_prob_list=[0.15]
rev_list=[0]
mlm_batch_list=[25]


if include_prior:
    dir_name='singlev0_prior_seed{}_rep1'.format(seed)
else:
    dir_name='singlev0_noprior_seed{}_rep1'.format(seed)
train_log_dir='logs/ti_models/train/{}'.format(dir_name)
exclude_cap_types='specific-human_interactions-creation'
# exclude_cap_types='specific-human_interactions-creation'
os.makedirs(train_log_dir,exist_ok=True) 
add_pe_list=[0,1]
for add_pe in add_pe_list:
    for mlm_batch in mlm_batch_list:
        for rev in rev_list:
            for mask_prob in mask_prob_list:
                mask_prob_str=float_to_str(mask_prob)
                mask_prob_str=mask_prob_str.replace('.','')
                for cidx,concept in enumerate(list(info_map.keys())):
                    device_idx=stat_idx
                    for lambda_mlm in lambda_mlm_list:
                        lambda_mlm_str=float_to_str(lambda_mlm)
                        lambda_mlm_str=lambda_mlm_str.replace('.','')
                        prior,category=info_map[concept]
                        prefix='tiv0_'
                        if include_prior:
                            prefix+='prior'
                        else:
                            prefix+='noprior'
                        if lambda_mlm:
                            run_name="{}_mlm{}_{}".format(prefix,lambda_mlm_str,concept)
                            run_name+='_mprob{}'.format(mask_prob_str)
                            run_name+='_mbatch{}'.format(mlm_batch)
                            run_name+='_cap4_bgoutfit'
                        else:
                            run_name="{}_nomlm_{}".format(prefix,concept)
                        if add_pe:
                            run_name+='_add_pe'
                        else:
                            run_name+='_no_pe'
                        output_dir=os.path.join('saved_models/ti_models/{}'.format(dir_name),concept)
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
                            print('training',run_name,'sleep','{}/{}'.format(cidx+1,len(concepts)))
                            time.sleep(delay)
                        print(run_name,device_idx)
                        log_path=os.path.join(train_log_dir,run_name+'.out')
                        command='export CUDA_VISIBLE_DEVICES={};'.format(device_idx)
                        command+='export CUBLAS_WORKSPACE_CONFIG=:4096:8;'
                        command+='accelerate launch --main_process_port {} train_mlm_single.py \\\n'.format(ports[cidx])
                        command+='--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \\\n'
                        command+='--train_data_dir1="/data/twkim/diffusion/personalization/collected/images/{}" \\\n'.format(concept)
                        command+='--learnable_property="object" \\\n'
                        command+='--placeholder_token1="<{}>" \\\n'.format(concept)
                        command+='--prior_concept1="{}" \\\n'.format(prior)
                        command+='--resolution=512 \\\n'
                        command+='--train_batch_size=1 \\\n'
                        command+='--gradient_accumulation_steps=4 \\\n'
                        command+='--max_train_steps=3001 \\\n'
                        command+='--learning_rate=5e-4 \\\n'
                        command+='--lr_scheduler="constant" \\\n'
                        command+='--add_pe={} \\\n'.format(add_pe)
                        command+='--lr_warmup_steps=0 \\\n'
                        command+='--output_dir="{}" \\\n'.format(output_dir)
                        command+='--seed={} \\\n'.format(seed)
                        command+='--mask_tokens="[MASK]" \\\n'
                        command+='--lambda_mlm={} --freeze_mask_embedding=1 \\\n'.format(lambda_mlm)
                        # command+='--cls_net_path="saved_models/mlm_models/sd1_contextnetv2_nonpadding_1e4/checkpoints/checkpoint-100000/cls_net_100000_ckpt.pt" \\\n'
                        # command+='--mask_embed_path="saved_models/mlm_models/sd1_contextnetv2_nonpadding_1e4/checkpoints/checkpoint-100000/mask_embeds_100000_ckpt.pt" \\\n'
                        command+='--cls_net_path="saved_models/mlm_models/mlm_contextnet_nonpad_lr1e4/checkpoints/cls_net_99000_ckpt.pt" \\\n'
                        command+='--mask_embed_path="saved_models/mlm_models/mlm_contextnet_nonpad_lr1e4/checkpoints/mask_embeds_99000_ckpt.pt" \\\n'
                        command+='--mask_prob={} \\\n'.format(mask_prob)
                        command+='--mlm_target=masked \\\n'
                        command+='--mlm_batch_size={} \\\n'.format(mlm_batch)
                        command+='--scale_lr \\\n'
                        command+='--prompt_type="{}" \\\n'.format(category)
                        command+='--silent=0 \\\n'
                        command+='--rev={} \\\n'.format(rev)
                        command+='--exclude_cap_types={} \\\n'.format(exclude_cap_types)
                        command+='--normalize_target1=0 \\\n'
                        command+='--caption_root="../datasets_pkgs/captions/v1" \\\n'
                        command+='--run_name="{}" \\\n'.format(run_name)
                        # command+='--report_to="wandb" \\\n'
                        # command+='--project_name="TI MLM SINGLE" \\\n'
                        command+='--include_prior_concept={} > {} 2>&1 &'.format(include_prior,log_path)

                        os.system(command)
                        print('TRAIN STARTED')
                        # break
                        time.sleep(delay)
        #             break
        #         break
        #     break
        # break
        
                


print('GENERATION')
# GENERATION
dir_path=os.path.join('saved_models/ti_models',dir_name)
gen_log_dir='logs/generate/{}'.format(dir_name)
target_step=3000
os.makedirs(gen_log_dir,exist_ok=True)    
delay=30
num_images_per_prompt=8
port_idx=0
include_prior_concept=1
ppos_list=[0]

for cidx,concept in enumerate(concepts):
    if concept not in info_map:
        continue
    concept_path=os.path.join(dir_path,concept)
    if not os.path.exists(concept_path):
        continue
    exps=os.listdir(concept_path)
    for ppos in ppos_list:
        ppos_str=float_to_str(ppos)
        ppos_str=ppos_str.replace('.','P')
        for exp_idx,exp in enumerate(exps):
            if '_rev' in exp:
                rev=1
            else:
                rev=0
            if '_add_pe' in exp:
                add_pe=1
            else:
                add_pe=0
            prior,category=info_map[concept]
            learned_embed_path1=os.path.join(concept_path,exp,'checkpoints/learned_embeds_s{}.pt'.format(target_step))
            if not os.path.exists(learned_embed_path1):
                print(learned_embed_path1,'does not exist')
                continue
            exp_name=exp
            exp_name+='_s{}'.format(target_step)
            exp_name+='_ppos{}'.format(ppos_str)
            output_dir=os.path.join('results_ti/{}/{}'.format(dir_name,concept))
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
                print('generating',exp,'sleep','{}/{}'.format(cidx+1,len(concepts)))
                time.sleep(delay)
            print(exp_name,device_idx)
            log_path=os.path.join(gen_log_dir,exp_name+'.out')
            command='export CUDA_VISIBLE_DEVICES={};'.format(device_idx)
            command+='export CUBLAS_WORKSPACE_CONFIG=:4096:8;'
            command+='accelerate launch --main_process_port {} generate_single.py \\\n'.format(ports[port_idx])
            command+='--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \\\n'
            command+='--train_data_dir1="/data/twkim/diffusion/personalization/collected/images/{}" \\\n'.format(concept)
            command+='--placeholder_token1="<{}>" \\\n'.format(concept)
            command+='--prior_concept1="{}" \\\n'.format(prior)
            command+='--resolution=512 \\\n'
            command+='--dst_exp_path={} \\\n'.format(exp_path)
            command+='--eval_batch_size=15 \\\n'
            command+='--add_pe={} \\\n'.format(add_pe)
            command+='--num_images_per_prompt={} \\\n'.format(num_images_per_prompt)
            command+='--output_dir="{}" \\\n'.format(output_dir)
            command+='--seed={} \\\n'.format(seed)
            command+='--mask_tokens="[MASK]" \\\n'
            command+='--learned_embed_path1="{}" \\\n'.format(learned_embed_path1)
            command+='--calibrate_ppos1="{}" \\\n'.format(ppos)
            command+='--rev={} \\\n'.format(rev)
            command+='--prompt_type="{}" \\\n'.format(category)
            command+='--include_prior_concept={} > {} 2>&1 &'.format(include_prior_concept,log_path)
            os.system(command)
            print('GENERATION STARTED')
            port_idx+=1
            time.sleep(delay)