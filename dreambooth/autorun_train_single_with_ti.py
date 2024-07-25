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
    'barn': ('barn','building'),
    'backpack':('backpack','nonliving'),
    'pet_dog1':('dog','pet'),
    'teddybear':('teddybear','nonliving'),
    'cat1': ('cat','pet'),
    'dog3': ('dog','pet'),
    'cat_statue': ('toy','nonliving'),
    'rc_car':('toy','nonliving'),
    'pink_sunglasses':('sunglasses','sunglasses'),
    'chair1': ('chair','nonliving'),
    # 'flower1':('flower','flower'),
    
}
# cuda_ids=[0,1,2,3,4,5,6,7]
lambda_mlms=[ 
             0.001,
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
log_dir='logs/train/single/with_ti'
os.makedirs(log_dir,exist_ok=True)   
ports=np.arange(1111,2222)
fixtes=[0]
pps=[1]
mlm_priors=[0]

# for idx,concept in enumerate(list(info_map.keys())):
for mlm_prior in mlm_priors:
    mlm_prior_str=float_to_str(mlm_prior)
    mlm_prior_str=mlm_prior_str.replace('.','')
    for lambda_mlm in lambda_mlms:
        device_idx=stat_idx
        for fixte in fixtes:
            for pp in pps:
                # for lambda_mlm in lambda_mlms:
                for idx,concept in enumerate(list(info_map.keys())):
                    lambda_mlm_str=float_to_str(lambda_mlm)
                    lambda_mlm_str=lambda_mlm_str.replace('.','')
                    prior,category=info_map[concept]
                    prefix='dreambooth_'
                    if pp:
                        run_name="{}pp".format(prefix)
                    else:
                        run_name="{}nopp".format(prefix)
                    if lambda_mlm:
                        run_name+="_mlm{}_{}".format(lambda_mlm_str,concept)
                    else:
                        run_name+="_nomlm_{}".format(concept)
                    if masked_loss:
                        run_name+='_masked'
                    if fixte:
                        run_name+='_fixte'
                    run_name+='_mlmprior{}'.format(mlm_prior_str)
                    run_name+='_with_ti'

                    output_dir=os.path.join('saved_models/dreambooth_models/single',concept)
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
                    learned_embed_path1='saved_models/ti_models/single_prior/{}/ti_norm0_prior_mlm0001_{}/checkpoints/learned_embeds_s3000.pt'.format(concept,concept)
                    log_path=os.path.join(log_dir,run_name+'.out')
                    command='export CUDA_VISIBLE_DEVICES={};'.format(device_idx)
                    command+='accelerate launch --main_process_port {} train_dreambooth_single_mlm.py \\\n'.format(ports[idx],idx)
                    command+='--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \\\n'
                    command+='--train_data_dir1="/data/twkim/diffusion/personalization/collected/images/{}" \\\n'.format(concept)
                    command+='--placeholder_token1="<{}>" \\\n'.format(concept)
                    command+='--prior_concept1="{}" \\\n'.format(prior)
                    command+='--resolution=512 \\\n'
                    command+='--train_batch_size=1 \\\n'
                    command+='--gradient_accumulation_steps=1 \\\n'
                    command+='--max_train_steps=1001 \\\n'
                    command+='--learning_rate=1e-6 \\\n'
                    command+='--lr_scheduler="constant" \\\n'
                    command+='--lr_warmup_steps=0 \\\n'
                    command+='--output_dir="{}" \\\n'.format(output_dir)
                    command+='--seed=7777 \\\n'
                    command+='--mask_tokens="[MASK]" \\\n'
                    command+='--lambda_mlm={} --freeze_mask_embedding=1 \\\n'.format(lambda_mlm)
                    command+='--cls_net_path="saved_models/mlm_contextnet_nonpad_lr1e4/checkpoints/cls_net_99000_ckpt.pt" \\\n'
                    command+='--mask_embed_path="saved_models/mlm_contextnet_nonpad_lr1e4/checkpoints/mask_embeds_99000_ckpt.pt" \\\n'
                    command+='--mlm_target=masked \\\n'
                    command+='--mlm_batch_size=20 \\\n'
                    command+='--prompt_type="{}" \\\n'.format(category)
                    command+='--silent=0 \\\n'
                    command+='--simple_caption=1 \\\n'
                    command+='--masked_loss={} \\\n'.format(masked_loss)
                    command+='--normalize_target1=0 \\\n'
                    command+='--mlm_prior={} \\\n'.format(mlm_prior)
                    command+='--run_name="{}" \\\n'.format(run_name)
                    command+='--with_prior_preservation={} \\\n'.format(pp)
                    command+='--class_prompt="a picture of a {}" \\\n'.format(prior)
                    command+='--learned_embed_path1="{}" \\\n'.format(learned_embed_path1)
                    command+='--class_data_dir="priors/{}" \\\n'.format(prior)
                    if not fixte:
                        command+='--train_text_encoder \\\n'
                    # command+='--report_to="wandb" \\\n'
                    # command+='--project_name="DreamBooth MLM SINGLE" \\\n'
                    command+='--include_prior_concept=1 > {} 2>&1 &'.format(log_path)
                    os.system(command)
                    print('STARTED')
                    time.sleep(20)
                


