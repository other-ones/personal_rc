import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from datasets_pkgs.mlm_pkgs.consts import BACKGROUNDS,STYLES,NONLIVINGS,NONLIVING_INTERACTIONS
from datasets_pkgs.mlm_pkgs.consts import HUMANS,RELATIVES,NONLIVING_ATTRIBUTES
from datasets_pkgs.mlm_pkgs.consts import ARTISTS,LOCATIONS,COLORS,FLOWER_INTERACTIONS,ANIMALS,FLOWER_ATTRIBUTES
from datasets_pkgs.mlm_pkgs.consts import ANIMAL_ATTRIBUTES


if __name__=='__main__':
    
    # out=caption_generator.generate_caption()
    # mlm_caption=caption_generator.generate_human_interactions_caption()
    # mlm_caption=caption_generator.generate_object_relations_caption()
    # mlm_caption=caption_generator.generate_styles_caption()
    # mlm_caption=caption_generator.generate_backgrounds_caption()
    # ['human_interactions','object_relations','backgrounds','styles']


    # 1. HUMAN INTERACTIONS
    captions_flower_interactions=[]
    for interaction in FLOWER_INTERACTIONS:
        for subject in HUMANS:
            prompt=f"<new1> is {interaction} a {subject}"
            captions_flower_interactions.append(prompt)

    
    # 2. object_relations
    captions_flower_relations=[]
    for rel in RELATIVES:
        for attr in NONLIVING_ATTRIBUTES:
            for other_obj in (NONLIVINGS):
                prompt=f"<new1> {rel} a {attr} {other_obj}"
                captions_flower_relations.append(prompt)
    for rel in RELATIVES:
        for attr in ANIMAL_ATTRIBUTES:
            for other_obj in (ANIMALS):
                prompt=f"<new1> {rel} a {attr} {other_obj}"
                captions_flower_relations.append(prompt)

    
    # 3. BACKGROUNDS no ACTION
    captions_flower_bg=[]
    for background in BACKGROUNDS:
        prompt1=f"<new1> with the {background} in the background"
        prompt2=f"A view of the <new1> at {background}"
        prompt3=f"<new1> captured at {background}"
        prompt4=f"<new1> growing at {background}"
        captions_flower_bg.append(prompt1)
        captions_flower_bg.append(prompt2)
        captions_flower_bg.append(prompt3)
        captions_flower_bg.append(prompt4)

    

    

    # 4. STYLE
    captions_flower_style=[]
    for style in STYLES:
        for fmt in ['captured','depicted','rendered',"painted"]:
            prompt1=f"<new1> {fmt} in the {style} style"
            prompt2=f"{style} painting of <new1>"
            captions_flower_style.append(prompt1)
            captions_flower_style.append(prompt2)
    captions_artist=[]
    for artist in ARTISTS:
        prompt3=f"painting of <new1> by artist {artist}"
        prompt4=f"<new1> painted by artist {artist}"
        captions_artist.append(prompt3)
        captions_artist.append(prompt4)
    captions_flower_style+=captions_artist


    
    
    # 5. ATTRIBUTES
    captions_flower_attr=[]
    for attr in FLOWER_ATTRIBUTES:
        prompt1=f"a {attr} <new1>"
        captions_flower_attr.append(prompt1)

    

    np.random.shuffle(captions_flower_interactions)
    np.random.shuffle(captions_flower_bg)
    np.random.shuffle(captions_flower_relations)
    np.random.shuffle(captions_flower_style)
    np.random.shuffle(captions_flower_attr)
    print('INTERACTIONS:',len(captions_flower_interactions),captions_flower_interactions[:5])
    print()
    print('RELATIVES:',len(captions_flower_relations),captions_flower_relations[:5])
    print()
    print('BG:',len(captions_flower_bg),captions_flower_bg[:5])
    print()
    print('STYLE:',len(captions_flower_style),captions_flower_style[:5])
    print()
    print('ATTR:',len(captions_flower_attr),captions_flower_attr[:5])
    print()


    dst_root='../../datasets_pkgs/captions/flower'
    os.makedirs(dst_root,exist_ok=True)
    dst_path=os.path.join(dst_root,'captions_flower_interactions.txt')
    dst_file=open(dst_path,'w')
    captions_flower_interactions=sorted(captions_flower_interactions)
    for caption in captions_flower_interactions:
        dst_file.write("{}\n".format(caption))
    
    dst_path=os.path.join(dst_root,'captions_flower_relations.txt')
    dst_file=open(dst_path,'w')
    captions_flower_relations=sorted(captions_flower_relations)
    for caption in captions_flower_relations:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_flower_bg.txt')
    dst_file=open(dst_path,'w')
    captions_flower_bg=sorted(captions_flower_bg)
    for caption in captions_flower_bg:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_flower_style.txt')
    dst_file=open(dst_path,'w')
    captions_flower_style=sorted(captions_flower_style)
    for caption in captions_flower_style:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_flower_attr.txt')
    dst_file=open(dst_path,'w')
    captions_flower_attr=sorted(captions_flower_attr)
    for caption in captions_flower_attr:
        dst_file.write("{}\n".format(caption))

    # # styles
# 


