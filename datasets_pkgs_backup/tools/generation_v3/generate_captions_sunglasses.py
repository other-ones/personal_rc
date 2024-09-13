import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from datasets_pkgs.mlm_pkgs.consts import BACKGROUNDS,STYLES,NONLIVINGS,NONLIVING_INTERACTIONS
from datasets_pkgs.mlm_pkgs.consts import HUMANS,RELATIVES,NONLIVING_ATTRIBUTES,ANIMAL_ATTRIBUTES
from datasets_pkgs.mlm_pkgs.consts import ARTISTS,LOCATIONS,COLORS,ANIMALS,SUNGLASSES_ACTIVITIES


if __name__=='__main__':

    # 1. HUMAN INTERACTIONS
    captions_sunglasses_human_interactions=[]
    for interaction in NONLIVING_INTERACTIONS:
        for subject in HUMANS:
            prompt=f"<new1> is {interaction} a {subject}"
            captions_sunglasses_human_interactions.append(prompt)

    # 2. object_relations
    captions_sunglasses_relations=[]
    for rel in RELATIVES:
        for attr in NONLIVING_ATTRIBUTES:
            for other_obj in NONLIVINGS:
                prompt=f"<new1> {rel} a {attr} {other_obj}"
                captions_sunglasses_relations.append(prompt)
    for rel in RELATIVES:
        for attr in ANIMAL_ATTRIBUTES:
            for other_obj in (ANIMALS):
                prompt=f"<new1> {rel} a {attr} {other_obj}"
                captions_sunglasses_relations.append(prompt)


    
    # 3. BACKGROUNDS no ACTION
    captions_sunglasses_bg=[]
    for background in BACKGROUNDS:
        prompt1=f"<new1> with the {background} in the background"
        prompt2=f"<new1> at {background}"
        prompt3=f"<new1> captured at {background}"
        captions_sunglasses_bg.append(prompt1)
        captions_sunglasses_bg.append(prompt2)
        captions_sunglasses_bg.append(prompt3)

    

    

    # 4. STYLE
    captions_sunglasses_style=[]
    for style in STYLES:
        for fmt in ['captured','depicted','rendered',"painted"]:
            prompt1=f"<new1> {fmt} in the {style} style"
            prompt2=f"{style} painting of <new1>"
            captions_sunglasses_style.append(prompt1)
            captions_sunglasses_style.append(prompt2)
    captions_artist=[]
    for artist in ARTISTS:
        prompt3=f"<new1> painting by artist {artist}"
        prompt4=f"<new1> painted by artist {artist}"
        captions_artist.append(prompt3)
        captions_artist.append(prompt4)
    captions_sunglasses_style+=captions_artist


    
    
    # 5. ATTRIBUTES
    captions_sunglasses_attr=[]
    for attr in NONLIVING_ATTRIBUTES:
        prompt1=f"a {attr} <new1>"
        captions_sunglasses_attr.append(prompt1)
        
    # 6. WEARING
    LIVINGS=ANIMALS+HUMANS
    captions_sunglasses_wearing=[]
    for subject in LIVINGS:
        prompt1=f"{subject} wearing <new1>"
        prompt2=f"<new1> worn by {subject}"
        captions_sunglasses_wearing.append(prompt1)
        captions_sunglasses_wearing.append(prompt2)
        for act in SUNGLASSES_ACTIVITIES:
            prompt3=f"{subject} wearing <new1> {act}"
            captions_sunglasses_wearing.append(prompt3)


    np.random.shuffle(captions_sunglasses_human_interactions)
    np.random.shuffle(captions_sunglasses_relations)
    np.random.shuffle(captions_sunglasses_style)
    np.random.shuffle(captions_sunglasses_attr)
    np.random.shuffle(captions_sunglasses_wearing)

    
    
    print('INTERACTIONS:',len(captions_sunglasses_human_interactions))
    for item in captions_sunglasses_human_interactions[:5]:
        print(item)
    print()
    print('RELATIVES:',len(captions_sunglasses_relations))
    for item in captions_sunglasses_relations[:5]:
        print(item)
    print()
    print('BG:',len(captions_sunglasses_bg))
    for item in captions_sunglasses_bg[:5]:
        print(item)
    print()
    print('STYLE:',len(captions_sunglasses_style))
    for item in captions_sunglasses_style[:5]:
        print(item)
    print()
    print('ATTR:',len(captions_sunglasses_attr))
    for item in captions_sunglasses_attr[:5]:
        print(item)
    print()
    print('WEAR:',len(captions_sunglasses_wearing))
    for item in captions_sunglasses_wearing[:5]:
        print(item)
    print()

    dst_root='../../datasets_pkgs/captions/sunglasses'
    os.makedirs(dst_root,exist_ok=True)
    dst_path=os.path.join(dst_root,'captions_sunglasses_human_interactions.txt')
    dst_file=open(dst_path,'w')
    captions_sunglasses_human_interactions=list(set(captions_sunglasses_human_interactions))
    captions_sunglasses_human_interactions=sorted(captions_sunglasses_human_interactions)
    for caption in captions_sunglasses_human_interactions:
        dst_file.write("{}\n".format(caption))
    
    dst_path=os.path.join(dst_root,'captions_sunglasses_relations.txt')
    dst_file=open(dst_path,'w')
    captions_sunglasses_relations=list(set(captions_sunglasses_relations))
    captions_sunglasses_relations=sorted(captions_sunglasses_relations)
    for caption in captions_sunglasses_relations:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_sunglasses_bg.txt')
    dst_file=open(dst_path,'w')
    captions_sunglasses_bg=list(set(captions_sunglasses_bg))
    captions_sunglasses_bg=sorted(captions_sunglasses_bg)
    for caption in captions_sunglasses_bg:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_sunglasses_style.txt')
    dst_file=open(dst_path,'w')
    captions_sunglasses_style=list(set(captions_sunglasses_style))
    captions_sunglasses_style=sorted(captions_sunglasses_style)
    for caption in captions_sunglasses_style:
        dst_file.write("{}\n".format(caption))


    dst_path=os.path.join(dst_root,'captions_sunglasses_attr.txt')
    dst_file=open(dst_path,'w')
    captions_sunglasses_attr=list(set(captions_sunglasses_attr))
    captions_sunglasses_attr=sorted(captions_sunglasses_attr)
    for caption in captions_sunglasses_attr:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_sunglasses_wearing.txt')
    dst_file=open(dst_path,'w')
    captions_sunglasses_wearing=list(set(captions_sunglasses_wearing))
    captions_sunglasses_wearing=sorted(captions_sunglasses_wearing)
    for caption in captions_sunglasses_wearing:
        dst_file.write("{}\n".format(caption))
    # # styles
# 


