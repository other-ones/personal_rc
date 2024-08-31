import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from consts_v6 import NONLIVINGS,COLORS,HUMANS,RELATIVES,STYLES,BACKGROUNDS
from consts_v6 import NONVLIVING_INTERACTIONS_PASSIVE,NONVLIVING_INTERACTIONS_ACTIVE
# from consts_v6 import ANIMALS
import shutil


if __name__=='__main__':
    dst_root='../../captions/v6/nonliving'
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
        print(dst_root,'deleted')
    os.makedirs(dst_root,exist_ok=True)
    
    """
    if sampled_type_pos=='human_interactions':
        anchor=self.generate_human_interactions_caption()
    elif sampled_type_pos=='object_relations':
        anchor=self.generate_object_relations_caption()
    elif sampled_type_pos=='backgrounds':
        anchor=self.generate_backgrounds_caption()
    elif sampled_type_pos=='styles':
        anchor=self.generate_styles_caption() 
    """
    
    # 1. HUMAN INTERACTIONS
    captions_nonliving_human_interactions=[]
    for interaction in NONVLIVING_INTERACTIONS_PASSIVE:
        for subject in (HUMANS):
            if subject[0].isupper():
                prompt=f"<new1> is {interaction} {subject}"
            else:
                if subject[0] in ['a','e','i','o','u']:
                    prompt=f"<new1> is {interaction} an {subject}"
                else:
                    prompt=f"<new1> is {interaction} a {subject}"
            captions_nonliving_human_interactions.append(prompt)

    for interaction in NONVLIVING_INTERACTIONS_ACTIVE:
        for subject in (HUMANS):
            if subject[0].isupper():
                prompt=f"{subject} is {interaction} <new1>"
            else:
                if subject[0] in ['a','e','i','o','u']:
                    prompt=f"an {subject} is {interaction} <new1>"
                else:
                    prompt=f"a {subject} is {interaction} <new1>"
            captions_nonliving_human_interactions.append(prompt)

     # 2. RELATIVES
    captions_nonliving_relations=[]
    for rel in RELATIVES:      
        for color in COLORS:
            for other_obj in NONLIVINGS:
                prompt=f"<new1> {rel} a {color} {other_obj}"
                captions_nonliving_relations.append(prompt)


    # 3. BACKGROUNDS
    captions_nonliving_backgrounds=[]
    for background in BACKGROUNDS:
        prompt=f"<new1> with the {background} in the background"
        captions_nonliving_backgrounds.append(prompt)

        prompt=f"<new1> at {background}"
        captions_nonliving_backgrounds.append(prompt)

        prompt=f"<new1> captured with the {background} in the background"
        captions_nonliving_backgrounds.append(prompt)

        prompt=f"<new1> viewed with the {background}"
        captions_nonliving_backgrounds.append(prompt)

    # 4. generate_styles_caption
    # fmt=np.random.choice(['captured','depicted','rendered'])
    # style=np.random.choice(styles)
    captions_nonliving_styles=[]
    for fmt in ['captured','depicted','rendered']:
        for style in STYLES:
            prompt=f"<new1> {fmt} in the {style} style"
            captions_nonliving_styles.append(prompt)

    

    np.random.shuffle(captions_nonliving_human_interactions)
    np.random.shuffle(captions_nonliving_relations)
    np.random.shuffle(captions_nonliving_backgrounds)
    np.random.shuffle(captions_nonliving_styles)
    captions_nonliving_relations=list(set(captions_nonliving_relations))
    captions_nonliving_human_interactions=list(set(captions_nonliving_human_interactions))
    captions_nonliving_backgrounds=list(set(captions_nonliving_backgrounds))
    captions_nonliving_styles=list(set(captions_nonliving_styles))
    print('HUMAN INTERACT:',len(captions_nonliving_human_interactions))
    for item in captions_nonliving_human_interactions[:5]:
        print(item)
    print()


    print('OBJ RELATIONS:',len(captions_nonliving_relations))
    for item in captions_nonliving_relations[:5]:
        print(item)
    # print()
    print('BACKGROUNDS:',len(captions_nonliving_backgrounds))
    for item in captions_nonliving_backgrounds[:5]:
        print(item)
    print()

    print('STYLES:',len(captions_nonliving_styles))
    for item in captions_nonliving_styles[:5]:
        print(item)
    print()


    
    dst_path=os.path.join(dst_root,'captions_nonliving_human_interactions.txt')
    dst_file=open(dst_path,'w')
    captions_nonliving_human_interactions=list(set(captions_nonliving_human_interactions))
    captions_nonliving_human_interactions=sorted(captions_nonliving_human_interactions)
    for caption in captions_nonliving_human_interactions:
        dst_file.write("{}\n".format(caption))
    
    dst_path=os.path.join(dst_root,'captions_nonliving_relations.txt')
    dst_file=open(dst_path,'w')
    captions_nonliving_relations=list(set(captions_nonliving_relations))
    captions_nonliving_relations=sorted(captions_nonliving_relations)
    for caption in captions_nonliving_relations:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_nonliving_backgrounds.txt')
    dst_file=open(dst_path,'w')
    captions_nonliving_backgrounds=list(set(captions_nonliving_backgrounds))
    captions_nonliving_backgrounds=sorted(captions_nonliving_backgrounds)
    for caption in captions_nonliving_backgrounds:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_nonliving_styles.txt')
    dst_file=open(dst_path,'w')
    captions_nonliving_styles=list(set(captions_nonliving_styles))
    captions_nonliving_styles=sorted(captions_nonliving_styles)
    for caption in captions_nonliving_styles:
        dst_file.write("{}\n".format(caption))

    


