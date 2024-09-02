import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from datasets_pkgs.mlm_pkgs.consts import BACKGROUNDS,STYLES,NONLIVINGS,SOLO_ACTIVITIES,HUMANS,RELATIVES,NONLIVING_ATTRIBUTES
from datasets_pkgs.mlm_pkgs.consts import ARTISTS,LOCATIONS,IN_OUTFITS,WEARINGS,COLORS,DOG_ATTRIBUTES,CREATIONS
from datasets_pkgs.mlm_pkgs.consts import CREATION_ATTRIBUTES
import shutil

if __name__=='__main__':
    dst_root='../../datasets_pkgs/captions/dog'
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
        print(dst_root,'deleted')
    os.makedirs(dst_root,exist_ok=True)

    # 1. BACKGROUNDS no ACTION
    captions_dog_bg_noact=[]
    for background in BACKGROUNDS:
        prompt=f"<new1> with the {background} in the background"
        prompt=f"<new1> captured at {background}"
        captions_dog_bg_noact.append(prompt)
        captions_dog_bg_noact.append(prompt)
    for location in LOCATIONS:
        prompt=f"<new1> {location}"
        captions_dog_bg_noact.append(prompt)


    # 2. BACKGROUNDS with ACTION
    captions_dog_bg_act=[]
    for location in LOCATIONS:
        for activity in SOLO_ACTIVITIES:
            prompt=f"<new1> {activity} is {location}"
            captions_dog_bg_act.append(prompt)
    

    # 3. STYLE
    captions_dog_style=[]
    for style in STYLES:
        for fmt in ['captured','depicted','rendered',"painted"]:
            prompt=f"<new1> {fmt} in the {style} style"
            captions_dog_style.append(prompt)
            prompt=f"{style} painting of <new1>"
            captions_dog_style.append(prompt)
    captions_artist=[]
    for artist in ARTISTS:
        prompt=f"<new1> painting by artist {artist}"
        captions_artist.append(prompt)
        prompt=f"<new1> painted by artist {artist}"
        captions_artist.append(prompt)
        # captions_artist.append(prompt3)
    captions_dog_style+=captions_artist


    # 4. RELATIVES
    captions_dog_relations=[]
    for rel in RELATIVES:
        for attr in NONLIVING_ATTRIBUTES:
            for other_obj in NONLIVINGS:
                prompt=f"<new1> {rel} a {attr} {other_obj}"
                captions_dog_relations.append(prompt)


    # # 5. HUMAN INTERACTIONS
    captions_dog_human_interactions=[]
    for act in SOLO_ACTIVITIES:
        for subject in HUMANS:
            if subject[0] in ['a','e','i','o','u']:
                prompt=f"<new1> is {act} with an {subject}"
            elif subject[0].isupper():
                prompt=f"<new1> is {act} with {subject}"
            else:
                prompt=f"<new1> is {act} with a {subject}"
            captions_dog_human_interactions.append(prompt)


    # 6. OUTFITS
    captions_dog_outfits=[]
    for outfit in IN_OUTFITS:
        for color in COLORS:
            prompt=f"<new1> in {color} {outfit}"
            captions_dog_outfits.append(prompt)
    for wearing in WEARINGS:
        for color in COLORS:
            prompt=f"<new1> wearing {color} {wearing}"
            captions_dog_outfits.append(prompt)
    captions_dog_act_outfits=[]
    for outfit in IN_OUTFITS:
        for act in SOLO_ACTIVITIES:
            prompt=f"<new1> in {outfit} is {act}"
            captions_dog_act_outfits.append(prompt)
    captions_dog_bg_outfits=[]
    for outfit in IN_OUTFITS:
        for location in LOCATIONS:
            prompt=f"<new1> in {outfit} {location}"
            captions_dog_bg_outfits.append(prompt)
    # 7. ATTRIBUTES
    captions_dog_attr=[]
    for attr in DOG_ATTRIBUTES:
        prompt=f"{attr} <new1>"
        captions_dog_attr.append(prompt)
    for attr in DOG_ATTRIBUTES:
        for location in LOCATIONS:
            prompt=f"{attr} <new1> {location}"
            captions_dog_attr.append(prompt)

    # 8. Creations
    captions_dog_creations=[]
    for creation in CREATIONS:
        for attr in CREATION_ATTRIBUTES:
            prompt=f"{attr} {creation} <new1>"
            captions_dog_creations.append(prompt)
    

    np.random.shuffle(captions_dog_bg_act)
    np.random.shuffle(captions_dog_bg_noact)
    np.random.shuffle(captions_dog_style)
    np.random.shuffle(captions_dog_relations)
    np.random.shuffle(captions_dog_human_interactions)
    np.random.shuffle(captions_dog_attr)
    np.random.shuffle(captions_dog_outfits)
    np.random.shuffle(captions_dog_act_outfits)
    np.random.shuffle(captions_dog_creations)
    np.random.shuffle(captions_dog_bg_outfits)
    print('BG NOACT:',len(captions_dog_bg_noact))
    for item in captions_dog_bg_noact[:5]:
        print(item)
    print()
    print('BG ACT:',len(captions_dog_bg_act))
    for item in captions_dog_bg_act[:5]:
        print(item)
    # print()
    print('STYLE:',len(captions_dog_style))
    for item in captions_dog_style[:5]:
        print(item)
    print()

    print('RELATIVES:',len(captions_dog_relations))
    for item in captions_dog_relations[:5]:
        print(item)
    print()

    print('INTERACTIONS:',len(captions_dog_human_interactions))
    for item in captions_dog_human_interactions[:5]:
        print(item)
    print()


    print('OUTFITS:',len(captions_dog_outfits))
    for item in captions_dog_outfits[:5]:
        print(item)
    print()

    print('ACT+OUTFITS:',len(captions_dog_act_outfits))
    for item in captions_dog_act_outfits[:5]:
        print(item)
    print()

    print('BG+OUTFITS:',len(captions_dog_bg_outfits))
    for item in captions_dog_act_outfits[:5]:
        print(item)
    print()


    print('ATTR:',len(captions_dog_attr))
    for item in captions_dog_attr[:5]:
        print(item)
    print()

    print('CREATIONS:',len(captions_dog_creations))
    for item in captions_dog_creations[:5]:
        print(item)
    print()

    
    dst_path=os.path.join(dst_root,'captions_dog_bg_noact.txt')
    dst_file=open(dst_path,'w')
    captions_dog_bg_noact=list(set(captions_dog_bg_noact))
    captions_dog_bg_noact=sorted(captions_dog_bg_noact)
    for caption in captions_dog_bg_noact:
        dst_file.write("{}\n".format(caption))
    
    dst_path=os.path.join(dst_root,'captions_dog_bg_act.txt')
    dst_file=open(dst_path,'w')
    captions_dog_bg_act=list(set(captions_dog_bg_act))
    captions_dog_bg_act=sorted(captions_dog_bg_act)
    for caption in captions_dog_bg_act:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_dog_style.txt')
    dst_file=open(dst_path,'w')
    captions_dog_style=list(set(captions_dog_style))
    captions_dog_style=sorted(captions_dog_style)
    for caption in captions_dog_style:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_dog_relations.txt')
    dst_file=open(dst_path,'w')
    captions_dog_relations=list(set(captions_dog_relations))
    captions_dog_relations=sorted(captions_dog_relations)
    for caption in captions_dog_relations:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_dog_human_interactions.txt')
    dst_file=open(dst_path,'w')
    captions_dog_human_interactions=list(set(captions_dog_human_interactions))
    captions_dog_human_interactions=sorted(captions_dog_human_interactions)
    for caption in captions_dog_human_interactions:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_dog_outfits.txt')
    dst_file=open(dst_path,'w')
    captions_dog_outfits=list(set(captions_dog_outfits))
    captions_dog_outfits=sorted(captions_dog_outfits)
    for caption in captions_dog_outfits:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_dog_act_outfits.txt')
    dst_file=open(dst_path,'w')
    captions_dog_act_outfits=list(set(captions_dog_act_outfits))
    captions_dog_act_outfits=sorted(captions_dog_act_outfits)
    for caption in captions_dog_act_outfits:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_dog_bg_outfits.txt')
    dst_file=open(dst_path,'w')
    captions_dog_bg_outfits=list(set(captions_dog_bg_outfits))
    captions_dog_bg_outfits=sorted(captions_dog_bg_outfits)
    for caption in captions_dog_bg_outfits:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_dog_attr.txt')
    dst_file=open(dst_path,'w')
    captions_dog_attr=list(set(captions_dog_attr))
    captions_dog_attr=sorted(captions_dog_attr)
    for caption in captions_dog_attr:
        dst_file.write("{}\n".format(caption))


    dst_path=os.path.join(dst_root,'captions_dog_creations.txt')
    dst_file=open(dst_path,'w')
    captions_dog_creations=list(set(captions_dog_creations))
    captions_dog_creations=[item.strip() for item in captions_dog_creations if item.strip()]
    captions_dog_creations=sorted(captions_dog_creations)
    for caption in captions_dog_creations:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_dog_specific.txt')
    dst_file=open(dst_path,'w')
    captions_dog_specific=open("../../datasets_pkgs/captions/specific_captions/cat_specific_captions.txt").readlines()
    captions_dog_specific=list(set(captions_dog_specific))
    captions_dog_specific=[item.strip() for item in captions_dog_specific if item.strip()]
    captions_dog_specific=sorted(captions_dog_specific)
    for caption in captions_dog_specific:
        dst_file.write("{}\n".format(caption))



