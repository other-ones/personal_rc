from utils import float_to_str
import time
import numpy as np
import os
import socket
hostname = socket.gethostname()
print(hostname,'hostname')
# concepts=os.listdir('/data/twkim/diffusion/personalization/collected/images')
info_map_03={
    # train_prior/eval_prior/train_prompt_type/eval_prompt_type
    'duck_toy':('duck','duck toy','nonliving','nonliving'),
    'dog6': ('dog','dog','pet','living'),
    'teapot':('teapot','teapot','nonliving','nonliving'),
    'cat1': ('cat','cat','pet','living'),

    'pet_cat1':('cat','cat','pet','living'),
    'wooden_pot':('pot','wooden pot','nonliving','nonliving'),
    'backpack_dog':('backpack','backpack','nonliving','nonliving'),
    'poop_emoji':('toy','toy','nonliving','nonliving'),
    'cat2':('cat','cat','pet','living'),
    'dog3':  ('dog','dog','pet','living'),
    'pet_dog1':('dog','dog','pet','living'),
    'backpack':('backpack','backpack','nonliving','nonliving'),
    'teddybear':('bear','teddy bear','nonliving','nonliving'),
    'cat_statue': ('toy','toy','nonliving','nonliving'),
    'rc_car':('toy','toy','nonliving','nonliving'),
    'chair1': ('chair','chair','nonliving','nonliving'),


    # NOT USED
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
    # 'dog6': ('dog','dog','pet','living'),
    # 'duck_toy':('duck','duck toy','nonliving','nonliving'),
    # 'pet_cat1':('cat','cat','pet','living'),

    # 'wooden_pot':('pot','wooden pot','nonliving','nonliving'),
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


    # UNUSED
    # 'red_cartoon':('character','cartoon character','pet','living'),
    # 'candle':('candle','candle','nonliving','nonliving'),
    # 'can':('can','can','nonliving','nonliving'),
    # 'pink_sunglasses':('sunglasses','sunglasses'),
    # 'barn': ('barn','barn'),
    # 'flower1':('flower','flower'),
}
if '03' in hostname:
    info_map=info_map_03
    host_suffix='03'
elif 'ubuntu' in hostname:
    info_map=info_map_01
    host_suffix='01'
elif '07' in hostname:
    info_map=info_map_03
    host_suffix='07'
elif '04' in hostname:
    info_map=info_map_03
    host_suffix='04'
else:
    assert False
concepts=list(info_map.keys())
# cuda_ids
# cuda_ids=[0,1,2,3,4,5,6,7]
lambda_mlm_list=[
            0,
            # 0.01,
            # 0.001,
            # 0.002,
            # 0.0001,
            # 0.0005,
            # 0.0002,
            0.0001,
            # 0.01,
            # 0.0002,
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
seed=7777
include_prior=0
delay=25
mask_prob_list=[0.25]
rev_list=[0]
rep_id=1
benchmark='dreambooth'
# mlm_target_list=['masked','non_special']
mlm_target_list=['masked']
nonmask_weight_list=[1]









train_batch_size=4
if train_batch_size==4:
    learning_rate='5e-4'
elif train_batch_size==1:
    learning_rate='1e-3'
else:
    assert False
if include_prior:
    dir_name='single_reduced{}_prior_seed{}_rep{}_qlab{}'.format(train_batch_size,seed,rep_id,host_suffix)
else:
    dir_name='single_reduced{}_noprior_seed{}_rep{}_qlab{}'.format(train_batch_size,seed,rep_id,host_suffix)
# exclude_cap_types='specific-human_interactions-creation'
# exclude_cap_types='specific-human_interactions-creation'RF
exclude_cap_types=None

train_steps=3001
mlm_batch=25
# check_tags=['VERB-ADJ-ADV-PROPN-ADP-NOUN','']
check_tags=['']
# check_tags=['']
for mask_prob in mask_prob_list:
    for nonmask_weight in nonmask_weight_list:
        nonmask_weight_str=float_to_str(nonmask_weight)
        nonmask_weight_str=nonmask_weight_str.replace('.','')
        for check_tag in check_tags:
            for mlm_target in mlm_target_list:
                mask_prob_str=float_to_str(mask_prob)
                mask_prob_str=mask_prob_str.replace('.','')
                for cidx,concept in enumerate(list(info_map.keys())):
                    device_idx=stat_idx
                    train_log_dir='logs/ti_models/train/{}/{}'.format(dir_name,concept)
                    os.makedirs(train_log_dir,exist_ok=True) 
                    for lambda_mlm in lambda_mlm_list:
                        lambda_mlm_str=float_to_str(lambda_mlm)
                        lambda_mlm_str=lambda_mlm_str.replace('.','')
                        train_prior,eval_prior,train_prompt_type,eval_prompt_type=info_map[concept]
                        prefix='ti_qlab{}'.format(host_suffix)
                        if include_prior:
                            prefix+='_prior'
                        else:
                            prefix+='_noprior'
                        if lambda_mlm:
                            run_name="{}_mlm{}_{}".format(prefix,lambda_mlm_str,concept)
                            run_name+='_mprob{}'.format(mask_prob_str)
                            run_name+='_mbatch{}'.format(mlm_batch)
                            
                            if mlm_target =='non_special':
                                run_name+='_mtarget_nonspec_nmw{}'.format(nonmask_weight_str)
                            elif mlm_target=='masked':
                                run_name+='_mtarget_masked'
                            if check_tag:
                                run_name+='_tagged'
                        else:
                            run_name="{}_nomlm_{}".format(prefix,concept)
                        
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
                            print('SLEEP TRAINING',run_name,'sleep','{}/{}'.format(cidx+1,len(concepts)))
                            time.sleep(delay)
                        print(dir_name,run_name,device_idx)
                        log_path=os.path.join(train_log_dir,run_name+'.out')
                        command='export CUDA_VISIBLE_DEVICES={};'.format(device_idx)
                        command+='export CUBLAS_WORKSPACE_CONFIG=:4096:8;'
                        command+='accelerate launch --main_process_port {} ti_train.py \\\n'.format(ports[cidx])
                        command+='--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \\\n'
                        command+='--train_data_dir1="/data/twkim/diffusion/personalization/collected/images/{}" \\\n'.format(concept)
                        command+='--learnable_property="object" \\\n'
                        command+='--placeholder_token1="<{}>" \\\n'.format(concept)
                        command+='--train_prior_concept1="{}" \\\n'.format(train_prior)
                        command+='--eval_prior_concept1="{}" \\\n'.format(eval_prior)
                        command+='--resolution=512 \\\n'
                        command+='--train_batch_size={} \\\n'.format(train_batch_size)
                        command+='--gradient_accumulation_steps=1 \\\n'
                        command+='--max_train_steps={} \\\n'.format(train_steps)
                        command+='--learning_rate={} \\\n'.format(learning_rate)
                        command+='--lr_scheduler="constant" \\\n'
                        command+='--initializer_token={} \\\n'.format(train_prior)
                        command+='--normalize_mask_embeds=0 \\\n'
                        # command+='--add_pe={} \\\n'.format(add_pe)
                        command+='--lr_warmup_steps=0 \\\n'
                        command+='--output_dir="{}" \\\n'.format(output_dir)
                        command+='--seed={} \\\n'.format(seed)
                        command+='--mask_tokens="[MASK]" \\\n'
                        command+='--lambda_mlm={} --freeze_mask_embedding=1 \\\n'.format(lambda_mlm)
                        command+='--cls_net_path="saved_models/mlm_models/sd1_contextnetv4_nonpadding_1e4_unnorm_mprob015_batch150/checkpoints/checkpoint-100000/cls_net_100000_ckpt.pt" \\\n'
                        command+='--mask_embed_path="saved_models/mlm_models/sd1_contextnetv4_nonpadding_1e4_unnorm_mprob015_batch150/checkpoints/checkpoint-100000/mask_embeds_100000_ckpt.pt" \\\n'
                        # command+='--cls_net_path="saved_models/mlm_models/sd1_contextnetv5_nonpadding_1e4_unnorm_mprob015_batch150/checkpoints/checkpoint-100000/cls_net_100000_ckpt.pt" \\\n'
                        # command+='--mask_embed_path="saved_models/mlm_models/sd1_contextnetv5_nonpadding_1e4_unnorm_mprob015_batch150/checkpoints/checkpoint-100000/mask_embeds_100000_ckpt.pt" \\\n'
                        command+='--mask_prob={} \\\n'.format(mask_prob)
                        command+='--mlm_batch_size={} \\\n'.format(mlm_batch)
                        command+='--scale_lr \\\n'
                        command+='--eval_prompt_type="{}" \\\n'.format(eval_prompt_type)
                        command+='--train_prompt_type="{}" \\\n'.format(train_prompt_type)
                        command+='--silent=0 \\\n'
                        command+='--rev=0 \\\n'
                        if exclude_cap_types is not None:
                            command+='--exclude_cap_types={} \\\n'.format(exclude_cap_types)
                        if check_tag:
                            command+='--check_tag={} \\\n'.format(check_tag)
                        command+='--mlm_target={} \\\n'.format(mlm_target)
                        if mlm_target!='masked':
                            command+='--nonmask_weight={} \\\n'.format(nonmask_weight)
                        command+='--normalize_target1=0 \\\n'
                        command+='--caption_root="../datasets_pkgs/captions/v7" \\\n'
                        command+='--run_name="{}" \\\n'.format(run_name)
                        command+='--report_to="wandb" \\\n'
                        command+='--project_name="TI MLM SINGLE" \\\n'
                        command+='--include_prior_concept={} > {} 2>&1 &'.format(include_prior,log_path)
                        os.system(command)
                        print('TRAIN STARTED')
                        time.sleep(delay)
        
                


print('GENERATION')
# GENERATION
dir_path=os.path.join('saved_models/ti_models',dir_name)

target_step=3000
delay=30
num_images_per_prompt=8
port_idx=0
include_prior_concept=1
exclude_key='mtarget_nonspec'
for cidx,concept in enumerate(concepts):
    if concept not in info_map:
        continue
    gen_log_dir='logs/generate/{}/{}'.format(dir_name,concept)
    os.makedirs(gen_log_dir,exist_ok=True)    
    concept_path=os.path.join(dir_path,concept)
    if not os.path.exists(concept_path):
        continue
    exps=os.listdir(concept_path)
    for exp_idx,exp in enumerate(exps):
        if '_rev' in exp:
            rev=1
        else:
            rev=0
        if exclude_key in exp:
            continue
        train_prior,eval_prior,train_prompt_type,eval_prompt_type=info_map[concept]
        learned_embed_path1=os.path.join(concept_path,exp,'checkpoints/learned_embeds_s{}.pt'.format(target_step))
        if not os.path.exists(learned_embed_path1):
            print(learned_embed_path1,'does not exist')
            continue
        exp_name=exp
        exp_name+='_s{}'.format(target_step)
        output_dir=os.path.join('results/ti_results_re/{}/{}'.format(dir_name,concept))
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
            print('SLEEP GENEARTING',exp,'sleep','{}/{}'.format(cidx+1,len(concepts)))
            time.sleep(delay)
        print(exp_name,device_idx)
        log_path=os.path.join(gen_log_dir,exp_name+'.out')
        command='export CUDA_VISIBLE_DEVICES={};'.format(device_idx)
        command+='export CUBLAS_WORKSPACE_CONFIG=:4096:8;'
        command+='accelerate launch --main_process_port {} ti_generate.py \\\n'.format(ports[port_idx])
        command+='--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \\\n'
        command+='--train_data_dir1="/data/twkim/diffusion/personalization/collected/images/{}" \\\n'.format(concept)
        command+='--placeholder_token1="<{}>" \\\n'.format(concept)
        command+='--train_prior_concept1="{}" \\\n'.format(train_prior)
        command+='--eval_prior_concept1="{}" \\\n'.format(eval_prior)
        command+='--resolution=512 \\\n'
        command+='--dst_exp_path={} \\\n'.format(exp_path)
        command+='--benchmark_path="../datasets_pkgs/eval_prompts/{}.json" \\\n'.format(benchmark)
        command+='--eval_batch_size=15 \\\n'
        # command+='--add_pe={} \\\n'.format(add_pe)
        command+='--num_images_per_prompt={} \\\n'.format(num_images_per_prompt)
        command+='--output_dir="{}" \\\n'.format(output_dir)
        command+='--seed={} \\\n'.format(seed)
        command+='--mask_tokens="[MASK]" \\\n'
        command+='--learned_embed_path1="{}" \\\n'.format(learned_embed_path1)
        command+='--calibrate_ppos1=0 \\\n'
        command+='--rev={} \\\n'.format(rev)
        command+='--train_prompt_type="{}" \\\n'.format(train_prompt_type)
        command+='--eval_prompt_type="{}" \\\n'.format(eval_prompt_type)
        command+='--include_prior_concept={} > {} 2>&1 &'.format(include_prior_concept,log_path)
        os.system(command)
        print('GENERATION STARTED')
        port_idx+=1
        time.sleep(delay)