import os
import numpy as np
import cv2

# bigger_reduced4_prior_seed7777_qlab03_rep1
root1='saved_models/pplus_models/bigger_reduced4_prior_seed7777_qlab03_rep1'
root2='saved_models/pplus_models/gclip_reduced4_prior_seed7777_qlab03_rep1'
concepts=os.listdir(root1)
for idx,concept in enumerate(concepts):
    print(f'{concept}\t{idx+1}/{len(concepts)}')
    concept_path1=os.path.join(root1,concept)
    concept_path2=os.path.join(root2,concept)
    if not os.path.exists(concept_path2):
        print(concept,'CONCEPT NOT EXISTS')
        continue
    for mlm in ['mlm0001','mlm00001','mlm00005','nomlm']:
        if mlm=='nomlm':
            # pplus_bigger_qlab03_prior_nomlm_cat1
            # pplus_gclip_qlab03_prior_nomlm_dog6
            exp_path1=os.path.join(concept_path1,f'pplus_bigger_qlab03_prior_{mlm}_{concept}')
            exp_path2=os.path.join(root2,concept,f'pplus_gclip_qlab03_prior_{mlm}_{concept}')
        else:
            # pplus_bigger_qlab03_prior_mlm00005_cat1_mprob015_mbatch12_mtarget_masked_midxs_none
            # pplus_gclip_qlab03_prior_mlm0001_dog6_mprob015_mbatch12_mtarget_masked_midxs_none
            exp_path1=os.path.join(concept_path1,f'pplus_bigger_qlab03_prior_{mlm}_{concept}_mprob015_mbatch12_mtarget_masked_midxs_none')
            exp_path2=os.path.join(root2,concept,f'pplus_gclip_qlab03_prior_{mlm}_{concept}_mprob015_mbatch12_mtarget_masked_midxs_none')
        if not (os.path.exists(exp_path1)):
            print(exp_path1)
            assert False
        if not os.path.exists(exp_path2):
            print(f'{exp_path2} not exists')
        # assert os.path.exists(exp_path1) and os.path.exists(exp_path2)
        sample_root1=os.path.join(exp_path1,'samples')
        sample_root2=os.path.join(exp_path2,'samples')
        flist=os.listdir(sample_root1)
        fpath1=os.path.join(sample_root1,'sample_00100.jpg')
        fpath2=os.path.join(sample_root2,'sample_00100.jpg')
        if not os.path.exists(fpath2):
            print(f'{exp_path2} not complete')
            continue
        img1=cv2.imread(fpath1)
        img2=cv2.imread(fpath2)
        if not np.all(img1==img2):
            print(exp_path1,'different')
            # print(exp_path2)
            print()
        # else:
        #     print('same')
    
    
        