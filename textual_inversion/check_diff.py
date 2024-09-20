import os
import numpy as np
import cv2

root1='/home/twkim/project/rich_context/textual_inversion/saved_models/ti_models/bigger_reduced4_prior_seed7777_qlab03_rep2'
root2='/home/twkim/project/rich_context/textual_inversion/saved_models/ti_models/gclip_reduced4_prior_seed7777_qlab03_rep2'
concepts=os.listdir(root1)
for idx,concept in enumerate(concepts):
    print(f'{concept}\t{idx+1}/{len(concepts)}')
    concept_path1=os.path.join(root1,concept)
    concept_path2=os.path.join(root2,concept)
    if not os.path.exists(concept_path2):
        print(concept,'not complete')
        continue
    for mlm in ['mlm0001','mlm00001','mlm00005','nomlm']:
        # ti_bigger_qlab03_prior_mlm0001_backpack_dog_mprob015_mbatch25_mtarget_masked
        # ti_gclip_qlab03_prior_mlm00005_backpack_dog_mprob015_mbatch25_mtarget_masked
        if mlm=='nomlm':
            # ti_bigger_qlab03_prior_nomlm_teapot
            exp_path1=os.path.join(concept_path1,f'ti_bigger_qlab03_prior_{mlm}_{concept}')
            exp_path2=os.path.join(root2,concept,f'ti_gclip_qlab03_prior_{mlm}_{concept}')
        else:
            exp_path1=os.path.join(concept_path1,f'ti_bigger_qlab03_prior_{mlm}_{concept}_mprob015_mbatch25_mtarget_masked')
            exp_path2=os.path.join(root2,concept,f'ti_gclip_qlab03_prior_{mlm}_{concept}_mprob015_mbatch25_mtarget_masked')
        if not (os.path.exists(exp_path1)):
            print(exp_path1)
            assert False
        if not os.path.exists(exp_path2):
            print(f'{exp_path2} not exists')
        # assert os.path.exists(exp_path1) and os.path.exists(exp_path2)
        sample_root1=os.path.join(exp_path1,'samples')
        sample_root2=os.path.join(exp_path2,'samples')
        flist=os.listdir(sample_root1)
        fpath1=os.path.join(sample_root1,'sample_03000.jpg')
        fpath2=os.path.join(sample_root2,'sample_03000.jpg')
        if not os.path.exists(fpath2):
            print(f'{exp_path2} not complete')
            continue
        img1=cv2.imread(fpath1)
        img2=cv2.imread(fpath2)
        if not np.all(img1==img2):
            print(exp_path1)
            print(exp_path2)
            print()
    
    
        