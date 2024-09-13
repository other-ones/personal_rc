import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from datasets_pkgs.mlm_pkgs.consts import BACKGROUNDS,STYLES,NONLIVINGS,NONLIVING_INTERACTIONS
from datasets_pkgs.mlm_pkgs.consts import HUMANS,RELATIVES,TEDDYBEAR_ATTRIBUTES
from datasets_pkgs.mlm_pkgs.consts import ARTISTS,LOCATIONS,COLORS,ANIMALS,ANIMAL_ATTRIBUTES,NONLIVING_MATERIALS,SOLO_ACTIVITIES


if __name__=='__main__':

    # 1. HUMAN INTERACTIONS
    captions_teddybear_human_interactions=[]
    for act in LIVING_INTERACTIONS:
        for subject in HUMANS:
            if subject[0] in ['a','e','i','o','u']:
                prompt=f"<new1> is {act} with an {subject}"
            elif subject[0].isupper():
                prompt=f"<new1> is {act} with {subject}"
            else:
                prompt=f"<new1> is {act} with a {subject}"
            captions_teddybear_human_interactions.append(prompt)
        
    # 2. object_relations
    captions_teddybear_object_relations=[]
    for rel in RELATIVES:
        for attr in TEDDYBEAR_ATTRIBUTES:
            for other_obj in NONLIVINGS:
                prompt=f"<new1> {rel} a {attr} {other_obj}"
                captions_teddybear_object_relations.append(prompt)
    captions_teddybear_object_relations=[]
    for rel in RELATIVES:
        for attr in ANIMAL_ATTRIBUTES:
            for other_obj in (ANIMALS):
                prompt=f"<new1> {rel} a {attr} {other_obj}"
                captions_teddybear_object_relations.append(prompt)

    
    # 3. BACKGROUNDS no ACTION
    captions_teddybear_bg=[]
    for background in BACKGROUNDS:
        prompt1=f"<new1> with the {background} in the background"
        prompt2=f"<new1> at {background}"
        prompt3=f"<new1> captured at {background}"
        captions_teddybear_bg.append(prompt1)
        captions_teddybear_bg.append(prompt2)
        captions_teddybear_bg.append(prompt3)
    # 2. BACKGROUNDS with ACTION
    for location in LOCATIONS:
        for activity in SOLO_ACTIVITIES:
            prompt3=f"<new1> {activity} {location}"
            captions_teddybear_bg.append(prompt3)

    
    # 4. STYLE
    captions_teddybear_style=[]
    for style in STYLES:
        for fmt in ['captured','depicted','rendered',"painted"]:
            prompt1=f"<new1> {fmt} in the {style} style"
            prompt2=f"{style} painting of <new1>"
            captions_teddybear_style.append(prompt1)
            captions_teddybear_style.append(prompt2)
    captions_artist=[]

    for artist in ARTISTS:
        prompt3=f"<new1> painting by artist {artist}"
        prompt4=f"<new1> painted by artist {artist}"
        captions_artist.append(prompt3)
        captions_artist.append(prompt4)
    captions_teddybear_style+=captions_artist
    
    # 5. ATTRIBUTES
    captions_teddybear_attr=[]
    for attr in TEDDYBEAR_ATTRIBUTES:
        prompt1=f"a {attr} <new1>"
        captions_teddybear_attr.append(prompt1)
    # 5. Materials
    for mat in NONLIVING_MATERIALS:
        prompt1=f"<new1> made of {mat}"
        captions_teddybear_attr.append(prompt1)

    np.random.shuffle(captions_teddybear_human_interactions)
    np.random.shuffle(captions_teddybear_object_relations)
    np.random.shuffle(captions_teddybear_bg)
    np.random.shuffle(captions_teddybear_style)
    np.random.shuffle(captions_teddybear_attr)
    num_prints=10
    print('INTERACTIONS:',len(captions_teddybear_human_interactions))
    for item in captions_teddybear_human_interactions[:num_prints]:
        print(item)
    print()
    print('OBJECT RELATIONS:',len(captions_teddybear_object_relations))
    for item in captions_teddybear_object_relations[:num_prints]:
        print(item)
    print()
    print('BG:',len(captions_teddybear_bg))
    for item in captions_teddybear_bg[:num_prints]:
        print(item)
    print()
    print('STYLE:',len(captions_teddybear_style))
    for item in captions_teddybear_style[:num_prints]:
        print(item)
    print()
    print('ATTR:',len(captions_teddybear_attr))
    for item in captions_teddybear_attr[:num_prints]:
        print(item)
    print()


    dst_root='../../datasets_pkgs/captions/teddybear'
    os.makedirs(dst_root,exist_ok=True)
    dst_path=os.path.join(dst_root,'captions_teddybear_human_interactions.txt')
    dst_file=open(dst_path,'w')
    captions_teddybear_human_interactions=list(set(captions_teddybear_human_interactions))
    captions_teddybear_human_interactions=sorted(captions_teddybear_human_interactions)
    for caption in captions_teddybear_human_interactions:
        dst_file.write("{}\n".format(caption))
    
    dst_path=os.path.join(dst_root,'captions_teddybear_object_relations.txt')
    dst_file=open(dst_path,'w')
    captions_teddybear_object_relations=list(set(captions_teddybear_object_relations))
    captions_teddybear_object_relations=sorted(captions_teddybear_object_relations)
    for caption in captions_teddybear_object_relations:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_teddybear_bg.txt')
    dst_file=open(dst_path,'w')
    captions_teddybear_bg=list(set(captions_teddybear_bg))
    captions_teddybear_bg=sorted(captions_teddybear_bg)
    for caption in captions_teddybear_bg:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_teddybear_style.txt')
    dst_file=open(dst_path,'w')
    captions_teddybear_style=list(set(captions_teddybear_style))
    captions_teddybear_style=sorted(captions_teddybear_style)
    for caption in captions_teddybear_style:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_teddybear_attr.txt')
    dst_file=open(dst_path,'w')
    captions_teddybear_attr=list(set(captions_teddybear_attr))
    captions_teddybear_attr=sorted(captions_teddybear_attr)
    for caption in captions_teddybear_attr:
        dst_file.write("{}\n".format(caption))

