import os
import numpy as np
import cv2
root1='saved_models/cd_models/sgpu_seed7777_qlab07_rep1'
root2='saved_models/cd_models/sgpu_seed7777_qlab07_rep2'
concepts=os.listdir(root2)
for concept in concepts:
    concept_path2=os.path.join(root2,concept)
    concept_path1=os.path.join(root1,concept)
    if not os.path.exists(concept_path1):
        # print(f"{concept_path1}\tCONCEPT NOT EXISTS\t{concept}")
        continue
    exps=os.listdir(concept_path2)
    for exp in exps:
        if '_ti' in exp:
            continue 
        exp_path2=os.path.join(concept_path2,exp)
        exp_path1=os.path.join(root1,concept,exp)
        if not os.path.exists(exp_path1):
            # print(f"{exp_path1}\tEXP NOT EXISTS\t{concept}")
            continue
        sample_path1=os.path.join(exp_path1,'samples/sample_00200.jpg')
        sample_path2=os.path.join(exp_path2,'samples/sample_00200.jpg')
        img1=cv2.imread(sample_path1)
        img2=cv2.imread(sample_path2)
        if not np.all(img1==img2):
            print(exp)
        print(exp_path2,'same')