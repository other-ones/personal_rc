import numpy as np
import time
import os
# concepts=os.listdir('/data/twkim/diffusion/personalization/collected/images')
info_map={
    ('pet_cat1','pet_dog1'):{
        'prior_concept1':'cat',
        'prior_concept2':'dog',
        'prompt_type':'two_pets',
    },
    # ('cat1','pet_dog1'):{
    #     'prior_concept1':'cat',
    #     'prior_concept2':'dog',
    #     'prompt_type':'two_pets',
    # },
    # ('dog6','pet_dog1'):{
    #     'prior_concept1':'dog',
    #     'prior_concept2':'dog',
    #     'prompt_type':'two_pets',
    # },
    # 'teddybear':('teddybear','nonliving'),
    # 'wooden_pot':('pot','nonliving'),
    # 'vase':('vase','nonliving'),
    # 'cat1': ('cat','pet'),
    # 'pet_cat1':('cat','pet'),
    # 'pet_dog1':('dog','pet'),
    # 'barn': ('barn','building'),

    # 'chair1': ('chair','nonliving'),
    # 'cat_statue': ('toy','nonliving'),
    # 'rc_car':('toy','nonliving'),
    # 'pink_sunglasses':('sunglasses','sunglasses'),
    # 'flower1':('flower','flower'),
    # 'dog3': ('dog','pet'),
    # 'dog6': ('dog','pet'),
}
concept_pairs=list(info_map.keys())


import subprocess as sp
def float_to_str(f):
    s = f"{f:.15f}"  # Start with a high precision
    return s.rstrip('0').rstrip('.') if '.' in s else s

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

dir_name='multi_train'
log_dir='logs/train/{}'.format(dir_name)
os.makedirs(log_dir,exist_ok=True)    
ports=np.arange(2111,3222)
stats=get_gpu_memory()
for stat_idx,stat in enumerate(stats):
    if stat>2e4:
        break
target_norms=[4,8,16,24]
lambda_mlm=0.001
# lambda_contrastives=[0.001,0.0001,0.00001]
lambda_contrastives=[0]
# sim_margins=[0]
sim_margins=[0.1,0,0.2,0.3,0.5,0.4]
lambda_sims=[0.01,0.001]
# lambda_sims=[0]
lambda_mlm_str=float_to_str(lambda_mlm)
lambda_mlm_str=lambda_mlm_str.replace('.','')
# dissim_layers_list=['1to12','all']
dissim_layers_list=['last']
for dissim_layers in dissim_layers_list:
    for target_norm in target_norms:
        for sim_margin in sim_margins:
            margin_str=float_to_str(sim_margin).replace('.','')
            for lambda_sim in lambda_sims:
                lambda_sim_str=float_to_str(lambda_sim).replace('.','')
                for lambda_contrastive in lambda_contrastives:
                    lambda_contrastive_str=float_to_str(lambda_contrastive)
                    lambda_contrastive_str=lambda_contrastive_str.replace('.','')
                    for idx,concept_pair in enumerate(concept_pairs):
                        print(concept_pair,'concept_pair')
                        meta_data=info_map[concept_pair]
                        concept1=concept_pair[0]
                        concept2=concept_pair[1]
                        prior_concept1=meta_data['prior_concept1']
                        prior_concept2=meta_data['prior_concept2']
                        prompt_type=meta_data['prompt_type']
                        prefix='ti_norm{}_multi_margin{}_sim{}'.format(target_norm,margin_str,lambda_sim_str)
                        if lambda_mlm:
                            run_name="{}_mlm{}_contrast{}_{}_{}".format(prefix,lambda_mlm_str,lambda_contrastive_str,concept1,concept2)
                            exp_name1="{}_mlm{}_{}".format(prefix,lambda_mlm_str,concept1)
                            exp_name2="{}_mlm{}_{}".format(prefix,lambda_mlm_str,concept2)
                        else:
                            run_name="{}_nomlm_contrast{}_{}_{}".format(prefix,lambda_contrastive_str,concept1,concept2)
                            exp_name1="{}_nomlm_{}".format(prefix,concept1)
                            exp_name2="{}_nomlm_{}".format(prefix,concept2)
                        run_name+='_dissim_{}'.format(dissim_layers)
                        # emb_path
                        learned_embed_path1="saved_models/single/{}/ti_norm{}_noprior_mlm0001_{}/checkpoints/learned_embeds_s3000.pt".format(concept1,target_norm,concept1)
                        learned_embed_path2="saved_models/single/{}/ti_norm{}_noprior_mlm0001_{}/checkpoints/learned_embeds_s3000.pt".format(concept2,target_norm,concept2)
                        assert os.path.exists(learned_embed_path1),learned_embed_path1
                        assert os.path.exists(learned_embed_path2),learned_embed_path2
                        # emb_path
                        
                        
                        
                        output_dir=os.path.join('saved_models/{}/{}_{}'.format(dir_name,concept1,concept2))
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
                            print(run_name,'sleep',stat_idx,stat,'remaining {}'.format(len(info_map)-idx))
                            time.sleep(15)
                            stat_idx+=1
                            stat_idx=(stat_idx%len(stats))
                        print(run_name,device_idx)
                        log_path=os.path.join(log_dir,run_name+'.out')
                        
                        command='export CUDA_VISIBLE_DEVICES={};'.format(device_idx)
                        command+='accelerate launch --main_process_port {} train_mlm_multi.py \\\n'.format(ports[idx])
                        command+='--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \\\n'
                        command+='--train_data_dir1="/data/twkim/diffusion/personalization/collected/images/{}" \\\n'.format(concept1)
                        command+='--train_data_dir2="/data/twkim/diffusion/personalization/collected/images/{}" \\\n'.format(concept2)
                        command+='--learnable_property="object" \\\n'
                        command+='--normalize_target1={} \\\n'.format(target_norm)
                        command+='--normalize_target2={} \\\n'.format(target_norm)
                        command+='--lambda_sim={} \\\n'.format(lambda_sim)
                        command+='--sim_margin={} \\\n'.format(sim_margin)
                        command+='--placeholder_token1="<{}>" \\\n'.format(concept1)
                        command+='--placeholder_token2="<{}>" \\\n'.format(concept2)
                        command+='--learned_embed_path1="{}" \\\n'.format(learned_embed_path1)
                        command+='--learned_embed_path2="{}" \\\n'.format(learned_embed_path2)
                        command+='--prior_concept1="{}" \\\n'.format(prior_concept1)
                        command+='--prior_concept2="{}" \\\n'.format(prior_concept2)
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
                        command+='--lambda_contrastive={} \\\n'.format(lambda_contrastive)
                        command+='--cls_net_path="saved_models/mlm_contextnet_nonpad_lr1e4/checkpoints/cls_net_99000_ckpt.pt" \\\n'
                        command+='--mask_embed_path="saved_models/mlm_contextnet_nonpad_lr1e4/checkpoints/mask_embeds_99000_ckpt.pt" \\\n'
                        command+='--mlm_target=masked \\\n'
                        command+='--mlm_batch_size=50 \\\n'
                        command+='--run_name="{}" \\\n'.format(run_name)
                        command+='--scale_lr \\\n'
                        command+='--prompt_type="{}" \\\n'.format(prompt_type)
                        command+='--masked_loss=1 \\\n'
                        command+='--make_composition=1 \\\n'
                        command+='--silent=0 \\\n'
                        command+='--dissim_layers={} \\\n'.format(dissim_layers)
                        command+='--report_to="wandb" \\\n'
                        command+='--project_name="TI MULTI MARGIN" \\\n'
                        command+='--include_prior_concept=0 > {} 2>&1 &'.format(log_path)
                        os.system(command)
                        print('STARTED')
                        # exit()
                        time.sleep(15)

                

