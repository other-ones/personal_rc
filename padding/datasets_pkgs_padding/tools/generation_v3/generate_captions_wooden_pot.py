import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from datasets_pkgs.mlm_pkgs.consts import BACKGROUNDS,STYLES,NONLIVINGS,NONLIVING_INTERACTIONS
from datasets_pkgs.mlm_pkgs.consts import HUMANS,RELATIVES,WOODENPOT_ATTRIBUTES
from datasets_pkgs.mlm_pkgs.consts import ARTISTS,LOCATIONS,COLORS,ANIMALS,ANIMAL_ATTRIBUTES,NONLIVING_MATERIALS


if __name__=='__main__':

    # 1. HUMAN INTERACTIONS
    captions_wooden_pot_human_interactions=[]
    for interaction in NONLIVING_INTERACTIONS:
        for subject in HUMANS:
            prompt=f"<new1> is {interaction} a {subject}"
            captions_wooden_pot_human_interactions.append(prompt)
    # 2. object_relations
    captions_wooden_pot_object_relations=[]
    for rel in RELATIVES:
        for attr in WOODENPOT_ATTRIBUTES:
            for other_obj in NONLIVINGS:
                prompt=f"<new1> {rel} a {attr} {other_obj}"
                captions_wooden_pot_object_relations.append(prompt)
    captions_wooden_pot_object_relations=[]
    for rel in RELATIVES:
        for attr in ANIMAL_ATTRIBUTES:
            for other_obj in (ANIMALS):
                prompt=f"<new1> {rel} a {attr} {other_obj}"
                captions_wooden_pot_object_relations.append(prompt)

    
    # 3. BACKGROUNDS no ACTION
    captions_wooden_pot_bg=[]
    for background in BACKGROUNDS:
        prompt1=f"<new1> with the {background} in the background"
        prompt2=f"<new1> at {background}"
        prompt3=f"<new1> captured at {background}"
        captions_wooden_pot_bg.append(prompt1)
        captions_wooden_pot_bg.append(prompt2)
        captions_wooden_pot_bg.append(prompt3)

    
    # 4. STYLE
    captions_wooden_pot_style=[]
    for style in STYLES:
        for fmt in ['captured','depicted','rendered',"painted"]:
            prompt1=f"<new1> {fmt} in the {style} style"
            prompt2=f"{style} painting of <new1>"
            captions_wooden_pot_style.append(prompt1)
            captions_wooden_pot_style.append(prompt2)
    captions_artist=[]
    for artist in ARTISTS:
        prompt3=f"<new1> painting by artist {artist}"
        prompt4=f"<new1> painted by artist {artist}"
        captions_artist.append(prompt3)
        captions_artist.append(prompt4)
    captions_wooden_pot_style+=captions_artist
    
    # 5. ATTRIBUTES
    captions_wooden_pot_attr=[]
    for attr in WOODENPOT_ATTRIBUTES:
        prompt1=f"a {attr} <new1>"
        captions_wooden_pot_attr.append(prompt1)
    for attr in WOODENPOT_ATTRIBUTES:
        for rel in RELATIVES:
            for other_obj in (NONLIVINGS):
                prompt=f"{attr} <new1> {rel} a {other_obj}"
                captions_wooden_pot_attr.append(prompt)
    # 5. Materials
    for mat in NONLIVING_MATERIALS:
        prompt1=f"<new1> made of {mat}"
        captions_wooden_pot_attr.append(prompt1)

    np.random.shuffle(captions_wooden_pot_human_interactions)
    np.random.shuffle(captions_wooden_pot_object_relations)
    np.random.shuffle(captions_wooden_pot_bg)
    np.random.shuffle(captions_wooden_pot_style)
    np.random.shuffle(captions_wooden_pot_attr)
    num_print=10
    captions_wooden_pot_human_interactions=list(set(captions_wooden_pot_human_interactions))
    print('INTERACTIONS:',len(captions_wooden_pot_human_interactions))
    for item in captions_wooden_pot_human_interactions[:num_print]:
        print(item)
    print()
    captions_wooden_pot_object_relations=list(set(captions_wooden_pot_object_relations))
    print('OBJECT RELATIONS:',len(captions_wooden_pot_object_relations))
    for item in captions_wooden_pot_object_relations[:num_print]:
        print(item)
    print()
    captions_wooden_pot_bg=list(set(captions_wooden_pot_bg))
    print('BG:',len(captions_wooden_pot_bg))
    for item in captions_wooden_pot_bg[:num_print]:
        print(item)
    print()
    captions_wooden_pot_style=list(set(captions_wooden_pot_style))
    print('STYLE:',len(captions_wooden_pot_style))
    for item in captions_wooden_pot_style[:num_print]:
        print(item)
    print()
    captions_wooden_pot_attr=list(set(captions_wooden_pot_attr))
    print('ATTR:',len(captions_wooden_pot_attr))
    for item in captions_wooden_pot_attr[:num_print]:
        print(item)
    print()


    dst_root='../../datasets_pkgs/captions/wooden_pot'
    os.makedirs(dst_root,exist_ok=True)
    dst_path=os.path.join(dst_root,'captions_wooden_pot_human_interactions.txt')
    dst_file=open(dst_path,'w')
    captions_wooden_pot_human_interactions=sorted(captions_wooden_pot_human_interactions)
    for caption in captions_wooden_pot_human_interactions:
        dst_file.write("{}\n".format(caption))
    
    dst_path=os.path.join(dst_root,'captions_wooden_pot_object_relations.txt')
    dst_file=open(dst_path,'w')
    captions_wooden_pot_object_relations=sorted(captions_wooden_pot_object_relations)
    for caption in captions_wooden_pot_object_relations:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_wooden_pot_bg.txt')
    dst_file=open(dst_path,'w')
    captions_wooden_pot_bg=sorted(captions_wooden_pot_bg)
    for caption in captions_wooden_pot_bg:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_wooden_pot_style.txt')
    dst_file=open(dst_path,'w')
    captions_wooden_pot_style=sorted(captions_wooden_pot_style)
    for caption in captions_wooden_pot_style:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_wooden_pot_attr.txt')
    dst_file=open(dst_path,'w')
    captions_wooden_pot_attr=sorted(captions_wooden_pot_attr)
    for caption in captions_wooden_pot_attr:
        dst_file.write("{}\n".format(caption))
    dst_path=os.path.join(dst_root,'captions_wooden_pot_specific.txt')
    dst_file=open(dst_path,'w')
    captions_wooden_pot_specific=open("../../datasets_pkgs/captions/specific_captions/wooden_pot_specific_captions.txt").readlines()
    captions_wooden_pot_specific=list(set(captions_wooden_pot_specific))
    captions_wooden_pot_specific=[item.strip() for item in captions_wooden_pot_specific if item.strip()]
    captions_wooden_pot_specific=sorted(captions_wooden_pot_specific)
    for caption in captions_wooden_pot_specific:
        dst_file.write("{}\n".format(caption))

