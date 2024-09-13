import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from consts_v6 import NONLIVINGS,COLORS,HUMANS,RELATIVES,STYLES,BACKGROUNDS
from consts_v6 import NONVLIVING_INTERACTIONS_PASSIVE,NONVLIVING_INTERACTIONS_ACTIVE
from consts_v6 import ANIMALS
import shutil


if __name__=='__main__':
    dst_root='../../captions/v6/nonliving'
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
        print(dst_root,'deleted')
    os.makedirs(dst_root,exist_ok=True)
    
    """
    if sampled_type_pos=='subject_interactions':
        anchor=self.generate_subject_interactions_caption()
    elif sampled_type_pos=='object_relations':
        anchor=self.generate_object_relations_caption()
    elif sampled_type_pos=='backgrounds':
        anchor=self.generate_backgrounds_caption()
    elif sampled_type_pos=='styles':
        anchor=self.generate_styles_caption() 
    """
    
    # 1. HUMAN INTERACTIONS
    captions_sunglasses_subject_interactions=[]
    for subject in (HUMANS+ANIMALS):
        if subject[0].isupper():
            prompt1=f"<new1> worn by {subject}"
            prompt2=f"<new1> is worn by {subject}"
        else:
            if subject[0] in ['a','e','i','o','u']:
                prompt1=f"<new1> worn by an {subject}"
                prompt2=f"<new1> is worn by an {subject}"
            else:
                prompt1=f"<new1> worn by a {subject}"
                prompt2=f"<new1> is worn by a {subject}"
        captions_sunglasses_subject_interactions.append(prompt1)
        captions_sunglasses_subject_interactions.append(prompt2)
        
    
    for subject in (HUMANS+ANIMALS):
        if subject[0].isupper():
            prompt1=f"{subject} is wearing <new1>"
            prompt2=f"{subject} wearing <new1>"
        else:
            if subject[0] in ['a','e','i','o','u']:
                prompt1=f"an {subject} wearing <new1>"
                prompt2=f"an {subject} is wearing <new1>"
            else:
                prompt1=f"a {subject} wearing <new1>"
                prompt2=f"a {subject} is wearing <new1>"
        captions_sunglasses_subject_interactions.append(prompt1)
        captions_sunglasses_subject_interactions.append(prompt2)

     # 2. RELATIVES
    captions_sunglasses_relations=[]
    for rel in RELATIVES:      
        for color in COLORS:
            for other_obj in NONLIVINGS:
                prompt=f"<new1> {rel} a {color} {other_obj}"
                captions_sunglasses_relations.append(prompt)


    # 3. BACKGROUNDS
    captions_sunglasses_backgrounds=[]
    for background in BACKGROUNDS:
        prompt=f"<new1> with the {background} in the background"
        captions_sunglasses_backgrounds.append(prompt)

        prompt=f"<new1> at {background}"
        captions_sunglasses_backgrounds.append(prompt)

        prompt=f"<new1> captured with the {background} in the background"
        captions_sunglasses_backgrounds.append(prompt)

        prompt=f"<new1> viewed with the {background}"
        captions_sunglasses_backgrounds.append(prompt)

    # 4. generate_styles_caption
    # fmt=np.random.choice(['captured','depicted','rendered'])
    # style=np.random.choice(styles)
    captions_sunglasses_styles=[]
    for fmt in ['captured','depicted','rendered']:
        for style in STYLES:
            prompt=f"<new1> {fmt} in the {style} style"
            captions_sunglasses_styles.append(prompt)

    

    np.random.shuffle(captions_sunglasses_subject_interactions)
    np.random.shuffle(captions_sunglasses_relations)
    np.random.shuffle(captions_sunglasses_backgrounds)
    np.random.shuffle(captions_sunglasses_styles)
    captions_sunglasses_relations=list(set(captions_sunglasses_relations))
    captions_sunglasses_subject_interactions=list(set(captions_sunglasses_subject_interactions))
    captions_sunglasses_backgrounds=list(set(captions_sunglasses_backgrounds))
    captions_sunglasses_styles=list(set(captions_sunglasses_styles))
    print('SUBJECT INTERACT:',len(captions_sunglasses_subject_interactions))
    for item in captions_sunglasses_subject_interactions[:5]:
        print(item)
    print()


    print('OBJ RELATIONS:',len(captions_sunglasses_relations))
    for item in captions_sunglasses_relations[:5]:
        print(item)
    print()

    print('BACKGROUNDS:',len(captions_sunglasses_backgrounds))
    for item in captions_sunglasses_backgrounds[:5]:
        print(item)
    print()

    print('STYLES:',len(captions_sunglasses_styles))
    for item in captions_sunglasses_styles[:5]:
        print(item)
    print()


    
    dst_path=os.path.join(dst_root,'captions_sunglasses_subject_interactions.txt')
    dst_file=open(dst_path,'w')
    captions_sunglasses_subject_interactions=list(set(captions_sunglasses_subject_interactions))
    captions_sunglasses_subject_interactions=sorted(captions_sunglasses_subject_interactions)
    for caption in captions_sunglasses_subject_interactions:
        dst_file.write("{}\n".format(caption))
    
    dst_path=os.path.join(dst_root,'captions_sunglasses_relations.txt')
    dst_file=open(dst_path,'w')
    captions_sunglasses_relations=list(set(captions_sunglasses_relations))
    captions_sunglasses_relations=sorted(captions_sunglasses_relations)
    for caption in captions_sunglasses_relations:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_sunglasses_backgrounds.txt')
    dst_file=open(dst_path,'w')
    captions_sunglasses_backgrounds=list(set(captions_sunglasses_backgrounds))
    captions_sunglasses_backgrounds=sorted(captions_sunglasses_backgrounds)
    for caption in captions_sunglasses_backgrounds:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_sunglasses_styles.txt')
    dst_file=open(dst_path,'w')
    captions_sunglasses_styles=list(set(captions_sunglasses_styles))
    captions_sunglasses_styles=sorted(captions_sunglasses_styles)
    for caption in captions_sunglasses_styles:
        dst_file.write("{}\n".format(caption))

    


