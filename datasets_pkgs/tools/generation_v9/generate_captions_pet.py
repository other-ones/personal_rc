import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from consts_v9 import HUMAN_INTERACTIONS,NONLIVINGS,COLORS,HUMANS,RELATIVES,STYLES,BACKGROUNDS
from consts_v9 import OUTFITS,WEARINGS
from consts_v9 import NONVLIVING_INTERACTIONS_ACTIVE,SHAPES,CREATIVES
import shutil


if __name__=='__main__':
    dst_root='../../captions/v9/pet'
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
        print(dst_root,'deleted')
    os.makedirs(dst_root,exist_ok=True)
    print(dst_root,'dst_root')
    
    # 1. HUMAN INTERACTIONS
    captions_pet_human_interactions=[]
    for interaction in HUMAN_INTERACTIONS:
        for subject in HUMANS:
            if subject[0].isupper():
                prompt=f"{subject} is {interaction} <new1>"
            else:
                if subject[0] in ['a','e','i','o','u']:
                    prompt=f"an {subject} is {interaction} <new1>"
                else:
                    prompt=f"a {subject} is {interaction} <new1>"
            captions_pet_human_interactions.append(prompt)
    

     # 2. RELATIVES
    captions_pet_relations=[]
    for rel in RELATIVES:      
        for color in COLORS:
            for other_obj in (NONLIVINGS):
                prompt=f"<new1> {rel} a {color} {other_obj}"
                captions_pet_relations.append(prompt)

    # background=np.random.choice(backgrounds)
    # if np.random.rand()<0.5:
    #     prompt=f"<new1> with the {background} in the background"
    # else:
    #     prompt=f"A view of the <new1> at {background}"

    # 3. BACKGROUNDS
    captions_pet_backgrounds=[]
    for background in BACKGROUNDS:
        prompt=f"<new1> with the {background} in the background"
        captions_pet_backgrounds.append(prompt)

        prompt=f"<new1> at {background}"
        captions_pet_backgrounds.append(prompt)

        prompt=f"<new1> captured with the {background} in the background"
        captions_pet_backgrounds.append(prompt)

        prompt=f"<new1> viewed with the {background}"
        captions_pet_backgrounds.append(prompt)

    # 4. generate_styles_caption
    # fmt=np.random.choice(['captured','depicted','rendered'])
    # style=np.random.choice(styles)
    captions_pet_styles=[]
    for fmt in ['captured','depicted','rendered']:
        for style in STYLES:
            prompt=f"<new1> {fmt} in the {style} style"
            captions_pet_styles.append(prompt)

    # 5. WEARINGS
    captions_pet_wearings=[]
    for wearing in WEARINGS:
        for color in COLORS:
            prompt=f"<new1> wearing a {color} {wearing}"
            captions_pet_wearings.append(prompt)
    for outfit in OUTFITS:
        for color in COLORS:
            prompt=f"<new1> in a {color} {outfit}"
            captions_pet_wearings.append(prompt)
            prompt=f"<new1> dressed in a {color} {outfit}"
            captions_pet_wearings.append(prompt)
    # # CREATIVES
    # captions_pet_creatives=[]
    # for creative in CREATIVES:
    #     prompt=f"a {creative} <new1>"
    #     captions_pet_creatives.append(prompt)
    # for color in COLORS:
    #     prompt1=f"a {color} <new1>"
    #     prompt2=f"a {color} colored <new1>"
    #     captions_pet_creatives.append(prompt1)
    #     captions_pet_creatives.append(prompt2)
    # for shape in SHAPES:
    #     prompt1=f"a {shape} shaped <new1>"
    #     prompt2=f"a <new1> in a {shape} shape"
    #     captions_pet_creatives.append(prompt1)
    #     captions_pet_creatives.append(prompt2)
    
    np.random.shuffle(captions_pet_human_interactions)
    np.random.shuffle(captions_pet_relations)
    np.random.shuffle(captions_pet_backgrounds)
    np.random.shuffle(captions_pet_styles)
    np.random.shuffle(captions_pet_wearings)
    captions_pet_relations=list(set(captions_pet_relations))
    captions_pet_human_interactions=list(set(captions_pet_human_interactions))
    captions_pet_backgrounds=list(set(captions_pet_backgrounds))
    captions_pet_styles=list(set(captions_pet_styles))
    captions_pet_wearings=list(set(captions_pet_wearings))
    # captions_pet_creatives=list(set(captions_pet_creatives))
    print('HUMAN INTERACT:',len(captions_pet_human_interactions))
    for item in captions_pet_human_interactions[:5]:
        print(item)
    print()

    print('OBJ RELATIONS:',len(captions_pet_relations))
    for item in captions_pet_relations[:5]:
        print(item)
    print()
    print('BACKGROUNDS:',len(captions_pet_backgrounds))
    for item in captions_pet_backgrounds[:5]:
        print(item)
    print()

    print('STYLES:',len(captions_pet_styles))
    for item in captions_pet_styles[:5]:
        print(item)
    print()

    print('WEARINGS:',len(set(captions_pet_wearings)))
    for item in captions_pet_wearings[:5]:
        print(item)
    print()

    # print('CREATIVES:',len(set(captions_pet_creatives)))
    # for item in captions_pet_creatives[:5]:
    #     print(item)
    # print()


    
    dst_path=os.path.join(dst_root,'captions_pet_human_interactions.txt')
    dst_file=open(dst_path,'w')
    captions_pet_human_interactions=list(set(captions_pet_human_interactions))
    captions_pet_human_interactions=sorted(captions_pet_human_interactions)
    for caption in captions_pet_human_interactions:
        dst_file.write("{}\n".format(caption))
    
    dst_path=os.path.join(dst_root,'captions_pet_relations.txt')
    dst_file=open(dst_path,'w')
    captions_pet_relations=list(set(captions_pet_relations))
    captions_pet_relations=sorted(captions_pet_relations)
    for caption in captions_pet_relations:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_pet_backgrounds.txt')
    dst_file=open(dst_path,'w')
    captions_pet_backgrounds=list(set(captions_pet_backgrounds))
    captions_pet_backgrounds=sorted(captions_pet_backgrounds)
    for caption in captions_pet_backgrounds:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_pet_styles.txt')
    dst_file=open(dst_path,'w')
    captions_pet_styles=list(set(captions_pet_styles))
    captions_pet_styles=sorted(captions_pet_styles)
    for caption in captions_pet_styles:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_pet_wearings.txt')
    dst_file=open(dst_path,'w')
    captions_pet_wearings=list(set(captions_pet_wearings))
    captions_pet_wearings=sorted(captions_pet_wearings)
    for caption in captions_pet_wearings:
        dst_file.write("{}\n".format(caption))


    # dst_path=os.path.join(dst_root,'captions_pet_creatives.txt')
    # dst_file=open(dst_path,'w')
    # captions_pet_creatives=list(set(captions_pet_creatives))
    # captions_pet_creatives=sorted(captions_pet_wearings)
    # for caption in captions_pet_creatives:
    #     dst_file.write("{}\n".format(caption))



