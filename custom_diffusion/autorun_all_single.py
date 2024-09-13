from utils import float_to_str,invert_scientific_notation
import time
import numpy as np
import os
import socket
hostname = socket.gethostname()
print(hostname,'hostname')
concepts=os.listdir('/data/twkim/diffusion/personalization/collected/images')
info_map={
    # train_prior/eval_prior/train_prompt_type/eval_prompt_type
    'duck_toy':('duck','duck toy','nonliving','nonliving'),
    # 'dog6': ('dog','dog','pet','living'),
    # 'teapot':('teapot','teapot','nonliving','nonliving'),
    # 'pet_cat1':('cat','cat','pet','living'),

    # 'wooden_pot':('pot','wooden pot','nonliving','nonliving'),
    # 'backpack_dog':('backpack','backpack','nonliving','nonliving'),
    # 'poop_emoji':('toy','toy','nonliving','nonliving'),
    # 'cat2':('cat','cat','pet','living'),
    # 'cat1': ('cat','cat','pet','living'),
    # 'dog3':  ('dog','dog','pet','living'),
    # 'pet_dog1':('dog','dog','pet','living'),
    # 'backpack':('backpack','backpack','nonliving','nonliving'),
    # 'cat_statue': ('toy','toy','nonliving','nonliving'),
    # 'rc_car':('toy','toy','nonliving','nonliving'),
    # 'chair1': ('chair','chair','nonliving','nonliving'),
    # 'teddybear':('bear','teddy bear','nonliving','nonliving'),

    
    
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
    target_devices=[0,1,2,3,4,5,6,7]
elif 'ubuntu' in hostname:
    target_devices=[0,1]
elif '04' in hostname:
    target_devices=[4,5,6,7]
lambda_mlm_list=[
            0.001,
            0, 
            0.0001,
            0.0005,
            # 0.00005,
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
mask_prob_list=[0.15]
seed=7777
rep_id=1
dir_name='single_capv7_seed{}_rep{}'.format(seed,rep_id)
log_dir='logs/train/{}'.format(dir_name)
os.makedirs(log_dir,exist_ok=True)   
lr_list=[1e-5]
mlm_batch_size=25
# ['VERB', 'ADJ','ADV','PROPN','ADP','NOUN']
check_tags=['']
# target_tags=''
for check_tag in check_tags:
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
                        run_name='cd_cnetv4'
                        if lambda_mlm:
                            run_name+="_mlm{}_{}".format(lambda_mlm_str,concept)
                        else:
                            run_name+="_nomlm_{}".format(concept)
                        if fixte:
                            run_name+='_fixte'
                        if lambda_mlm:
                            run_name+='_mprob{}'.format(mask_prob_str)
                            run_name+='_mbatch{}'.format(mlm_batch_size)
                        run_name+='_lr{}'.format(lr_str)
                        if check_tag:
                            run_name+='_tagged'
                        output_dir=os.path.join('saved_models/cd_models/{}'.format(dir_name),concept)
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
                        print(exp_path,device_idx)
                        log_path=os.path.join(log_dir,run_name+'.out')
                        command='export CUDA_VISIBLE_DEVICES={};'.format(device_idx)
                        command+='export CUBLAS_WORKSPACE_CONFIG=:4096:8;'
                        command+='accelerate launch --main_process_port {} cd_train.py \\\n'.format(ports[port_idx])
                        command+='--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \\\n'
                        command+='--train_data_dir1="/data/twkim/diffusion/personalization/collected/images/{}" \\\n'.format(concept)
                        command+='--initializer_token=sks \\\n'
                        command+='--placeholder_token1="<{}>" \\\n'.format(concept)
                        command+='--train_prior_concept1="{}" \\\n'.format(train_prior)
                        command+='--eval_prior_concept1="{}" \\\n'.format(eval_prior)
                        command+='--eval_prompt_type="{}" \\\n'.format(eval_prompt_type)
                        command+='--train_prompt_type="{}" \\\n'.format(train_prompt_type)
                        command+='--resolution=512 \\\n'
                        command+='--train_batch_size=1 \\\n'
                        command+='--gradient_accumulation_steps=1 \\\n'
                        command+='--max_train_steps=501 \\\n'
                        command+='--validation_steps=100 \\\n'
                        command+='--learning_rate={} \\\n'.format(lr)
                        command+='--lr_scheduler="constant" \\\n'
                        command+='--lr_warmup_steps=0 \\\n'
                        command+='--output_dir="{}" \\\n'.format(output_dir)
                        command+='--seed={} \\\n'.format(seed)
                        command+='--mask_tokens="[MASK]" \\\n'
                        command+='--lambda_mlm={} --freeze_mask_embedding=1 \\\n'.format(lambda_mlm)
                        command+='--cls_net_path="saved_models/mlm_models/sd1_contextnetv4_nonpadding_1e4_unnorm_mprob015_batch150/checkpoints/checkpoint-100000/cls_net_100000_ckpt.pt" \\\n'
                        command+='--mask_embed_path="saved_models/mlm_models/sd1_contextnetv4_nonpadding_1e4_unnorm_mprob015_batch150/checkpoints/checkpoint-100000/mask_embeds_100000_ckpt.pt" \\\n'
                        command+='--mlm_target=masked \\\n'
                        command+='--mlm_batch_size={} \\\n'.format(mlm_batch_size)
                        command+='--mask_prob={} \\\n'.format(mask_prob)
                        command+='--silent=0 \\\n'
                        command+='--scale_lr \\\n'
                        command+='--simple_caption=0 \\\n'
                        command+='--masked_loss={} \\\n'.format(masked_loss)
                        command+='--normalize_target1=0 \\\n'
                        command+='--run_name="{}" \\\n'.format(run_name)
                        command+='--with_prior_preservation={} \\\n'.format(1)
                        command+='--class_prompt1="a picture of a {}" \\\n'.format(train_prior)
                        command+='--class_data_dir1="priors/samples_{}" \\\n'.format(train_prior)
                        command+='--caption_root="../datasets_pkgs/captions/v7" \\\n'
                        if fixte==0: # do not fix text_encoder
                            command+='--train_text_encoder=1 \\\n'
                        # command+='--report_to="wandb" \\\n'
                        # command+='--project_name="DreamBooth MLM SINGLE" \\\n'
                        command+='--include_prior_concept=1 > {} 2>&1 &'.format(log_path)
                        os.system(command)
                        print('TRAIN STARTED')
                        # exit()
                        time.sleep(15)




print('GENERATION')
# GENERATION
dir_path=os.path.join('saved_models/cd_models',dir_name)
gen_log_dir='logs/generate/{}'.format(dir_name)
os.makedirs(gen_log_dir,exist_ok=True)    
delay=30
num_images_per_prompt=8
port_idx=0
include_prior_concept=1
ppos_list=[0]
benchmark='dreambooth'
concepts=list(info_map.keys())
concepts=sorted(concepts)
get_target_step=0

for concept in concepts:
    if concept not in info_map:
        continue
    concept_path=os.path.join(dir_path,concept)
    if not os.path.exists(concept_path):
        continue
    exps=os.listdir(concept_path)
    for exp_idx,exp in enumerate(exps):
        train_prior,eval_prior,train_prompt_type,eval_prompt_type=info_map[concept]
        resume_cd_path=os.path.join(concept_path,exp,'checkpoints/checkpoint-{}/custom_diffusion.pt'.format(get_target_step))
        learned_embed_path1=os.path.join(concept_path,exp,'checkpoints/checkpoint-{}/learned_embed.pt'.format(get_target_step))
        if not os.path.exists(resume_cd_path):
            print(resume_cd_path,'does not exist')
            continue
        exp_name=resume_cd_path.split('/')[-4]
        exp_name+='_s{}'.format(get_target_step)
        output_dir=os.path.join('results/cd_results/{}/{}'.format(dir_name,concept))
        dst_exp_path=os.path.join(output_dir,exp_name)
        if os.path.exists(dst_exp_path):
            print(dst_exp_path,'exists')
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
            print(exp_name,'sleep',stat_idx,stat)
            time.sleep(10)
        print(exp_name,device_idx)
        log_path=os.path.join(gen_log_dir,exp_name+'.out')
        command='export CUDA_VISIBLE_DEVICES={};'.format(device_idx)
        command+='export CUBLAS_WORKSPACE_CONFIG=:4096:8;'
        command+='accelerate launch --main_process_port {} cd_generate.py \\\n'.format(ports[port_idx])
        command+='--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \\\n'
        command+='--train_data_dir1="/data/twkim/diffusion/personalization/collected/images/{}" \\\n'.format(concept)
        command+='--placeholder_token1="<{}>" \\\n'.format(concept)
        command+='--resolution=512 \\\n'
        command+='--eval_batch_size=15 \\\n'
        command+='--num_images_per_prompt={} \\\n'.format(num_images_per_prompt)
        command+='--output_dir="{}" \\\n'.format(output_dir)
        command+='--seed={} \\\n'.format(seed)
        command+='--mask_tokens="[MASK]" \\\n'
        command+='--resume_cd_path="{}" \\\n'.format(resume_cd_path)
        command+='--learned_embed_path1="{}" \\\n'.format(learned_embed_path1)
        command+='--dst_exp_path="{}" \\\n'.format(dst_exp_path)
        command+='--train_prior_concept1="{}" \\\n'.format(train_prior)
        command+='--eval_prior_concept1="{}" \\\n'.format(eval_prior)
        command+='--train_prompt_type="{}" \\\n'.format(train_prompt_type)
        command+='--eval_prompt_type="{}" \\\n'.format(eval_prompt_type)
        command+='--benchmark_path="../datasets_pkgs/eval_prompts/{}.json" \\\n'.format(benchmark)
        command+='--include_prior_concept=1 > {} 2>&1 &'.format(log_path)
        os.system(command)
        print('STARTED')
        port_idx+=1
        time.sleep(30)

