import time
import numpy as np
import os
import socket

hostname = socket.gethostname()
print(hostname,'hostname')
exit()
concepts=os.listdir('/data/twkim/diffusion/personalization/collected/images')
info_map={
    'dog6': ('dog','pet'),
    'pet_cat1':('cat','pet'),
    'vase':('vase','nonliving'),
    'cat1': ('cat','pet'),
    'pet_dog1':('dog','pet'),
    'backpack':('backpack','nonliving'),
    'barn': ('barn','building'),
    'teddybear':('teddybear','nonliving'),
    'wooden_pot':('pot','nonliving'),

    'dog3': ('dog','pet'),
    'chair1': ('chair','nonliving'),
    'cat_statue': ('toy','nonliving'),

    # 'rc_car':('toy','nonliving'),
    # 'pink_sunglasses':('sunglasses','sunglasses'),


    # 'flower1':('flower','flower'),
    
}
# cuda_ids=[0,1,2,3,4,5,6,7]
lambda_mlms=[0,0.001]
target_norms=[0]
masked_loss=0


import subprocess as sp
def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values
def float_to_str(f):
    s = f"{f:.15f}"  # Start with a high precision
    return s.rstrip('0').rstrip('.') if '.' in s else s
stats=get_gpu_memory()
for stat_idx,stat in enumerate(stats):
    if stat>2e4:
        break
log_dir='logs/pplus/train/single'
os.makedirs(log_dir,exist_ok=True)   
ports=np.arange(1111,2222)


include_priors=[1]
for include_prior in include_priors:
    for lambda_mlm in lambda_mlms:
        lambda_mlm_str=float_to_str(lambda_mlm)
        lambda_mlm_str=lambda_mlm_str.replace('.','')
        device_idx=stat_idx
        for target_norm in target_norms:
            for idx,concept in enumerate(list(info_map.keys())):
                prior,category=info_map[concept]
                prefix='pplus_norm{}_'.format(target_norm)
                if include_prior:
                    prefix+='prior'
                else:
                    prefix+='noprior'
                if lambda_mlm:
                    run_name="{}_mlm{}_{}".format(prefix,lambda_mlm_str,concept)
                else:
                    run_name="{}_nomlm_{}".format(prefix,concept)
                if masked_loss:
                    run_name+='_masked'
                output_dir=os.path.join('saved_models/pplus_models/single',concept)
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
                command+='accelerate launch --main_process_port {} train_pplus_single.py \\\n'.format(ports[idx],idx)
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
                command+='--lr_warmup_steps=0 \\\n'
                command+='--output_dir="{}" \\\n'.format(output_dir)
                command+='--seed=7777 \\\n'
                command+='--mask_tokens="[MASK]" \\\n'
                command+='--lambda_mlm={} --freeze_mask_embedding=1 \\\n'.format(lambda_mlm)
                command+='--cls_net_path="saved_models/mlm_contextnet_nonpad_lr1e4/checkpoints/cls_net_99000_ckpt.pt" \\\n'
                command+='--mask_embed_path="saved_models/mlm_contextnet_nonpad_lr1e4/checkpoints/mask_embeds_99000_ckpt.pt" \\\n'
                command+='--mlm_target=masked \\\n'
                command+='--mlm_batch_size=15 \\\n'
                command+='--scale_lr \\\n'
                command+='--prompt_type="{}" \\\n'.format(category)
                command+='--silent=0 \\\n'
                command+='--masked_loss={} \\\n'.format(masked_loss)
                command+='--normalize_target1={} \\\n'.format(target_norm)
                command+='--run_name="{}" \\\n'.format(run_name)
                command+='--report_to="wandb" \\\n'
                command+='--num_vectors1=9 \\\n'
                command+='--project_name="PPLUS MLM SINGLE" \\\n'
                command+='--include_prior_concept={} > {} 2>&1 &'.format(include_prior,log_path)

                os.system(command)
                print('STARTED')
                time.sleep(15)
            


