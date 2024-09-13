import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from datasets_pkgs.mlm_pkgs.consts import BACKGROUNDS,STYLES,NONLIVINGS,NONLIVING_INTERACTIONS
from datasets_pkgs.mlm_pkgs.consts import HUMANS,RELATIVES,NONLIVING_ATTRIBUTES
from datasets_pkgs.mlm_pkgs.consts import ARTISTS,LOCATIONS,COLORS,ANIMALS,ANIMAL_ATTRIBUTES,VASE_MATERIALS


if __name__=='__main__':

    # 1. HUMAN INTERACTIONS
    captions_vase_human_interactions=[]
    for interaction in NONLIVING_INTERACTIONS:
        for subject in HUMANS:
            prompt=f"<new1> is {interaction} a {subject}"
            captions_vase_human_interactions.append(prompt)
    # 2. object_relations
    captions_vase_object_relations=[]
    for rel in RELATIVES:
        for attr in NONLIVING_ATTRIBUTES:
            for other_obj in NONLIVINGS:
                prompt=f"<new1> {rel} a {attr} {other_obj}"
                captions_vase_object_relations.append(prompt)
    captions_vase_object_relations=[]
    for rel in RELATIVES:
        for attr in ANIMAL_ATTRIBUTES:
            for other_obj in (ANIMALS):
                prompt=f"<new1> {rel} a {attr} {other_obj}"
                captions_vase_object_relations.append(prompt)

    
    # 3. BACKGROUNDS no ACTION
    captions_vase_bg=[]
    for background in BACKGROUNDS:
        prompt1=f"<new1> with the {background} in the background"
        prompt2=f"<new1> at {background}"
        prompt3=f"<new1> captured at {background}"
        captions_vase_bg.append(prompt1)
        captions_vase_bg.append(prompt2)
        captions_vase_bg.append(prompt3)

    
    # 4. STYLE
    captions_vase_style=[]
    for style in STYLES:
        for fmt in ['captured','depicted','rendered',"painted"]:
            prompt1=f"<new1> {fmt} in the {style} style"
            prompt2=f"{style} painting of <new1>"
            captions_vase_style.append(prompt1)
            captions_vase_style.append(prompt2)
    captions_artist=[]
    for artist in ARTISTS:
        prompt3=f"<new1> painting by artist {artist}"
        prompt4=f"<new1> painted by artist {artist}"
        captions_artist.append(prompt3)
        captions_artist.append(prompt4)
    captions_vase_style+=captions_artist
    
    # 5. ATTRIBUTES
    captions_vase_attr=[]
    for attr in NONLIVING_ATTRIBUTES:
        prompt1=f"a {attr} <new1>"
        captions_vase_attr.append(prompt1)
    # 5. Materials
    for mat in VASE_MATERIALS:
        prompt1=f"<new1> made of {mat}"
        captions_vase_attr.append(prompt1)

    np.random.shuffle(captions_vase_human_interactions)
    np.random.shuffle(captions_vase_object_relations)
    np.random.shuffle(captions_vase_bg)
    np.random.shuffle(captions_vase_style)
    np.random.shuffle(captions_vase_attr)
    print('INTERACTIONS:',len(captions_vase_human_interactions))
    for item in captions_vase_human_interactions[:5]:
        print(item)
    print()
    print('OBJECT RELATIONS:',len(captions_vase_object_relations))
    for item in captions_vase_object_relations[:5]:
        print(item)
    print()
    print('BG:',len(captions_vase_bg),captions_vase_bg[:5])
    for item in captions_vase_bg[:5]:
        print(item)
    print()
    print('STYLE:',len(captions_vase_style))
    for item in captions_vase_style[:5]:
        print(item)
    print()
    print('ATTR:',len(captions_vase_attr))
    for item in captions_vase_attr[:5]:
        print(item)
    print()


    dst_root='../../datasets_pkgs/captions/vase'
    os.makedirs(dst_root,exist_ok=True)
    dst_path=os.path.join(dst_root,'captions_vase_human_interactions.txt')
    dst_file=open(dst_path,'w')
    captions_vase_human_interactions=list(set(captions_vase_human_interactions))
    captions_vase_human_interactions=sorted(captions_vase_human_interactions)
    for caption in captions_vase_human_interactions:
        dst_file.write("{}\n".format(caption))
    
    dst_path=os.path.join(dst_root,'captions_vase_object_relations.txt')
    dst_file=open(dst_path,'w')
    captions_vase_object_relations=list(set(captions_vase_object_relations))
    captions_vase_object_relations=sorted(captions_vase_object_relations)
    for caption in captions_vase_object_relations:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_vase_bg.txt')
    dst_file=open(dst_path,'w')
    captions_vase_bg=list(set(captions_vase_bg))
    captions_vase_bg=sorted(captions_vase_bg)
    for caption in captions_vase_bg:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_vase_style.txt')
    dst_file=open(dst_path,'w')
    captions_vase_style=list(set(captions_vase_style))
    captions_vase_style=sorted(captions_vase_style)
    for caption in captions_vase_style:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_vase_attr.txt')
    dst_file=open(dst_path,'w')
    captions_vase_attr=list(set(captions_vase_attr))
    captions_vase_attr=sorted(captions_vase_attr)
    for caption in captions_vase_attr:
        dst_file.write("{}\n".format(caption))

