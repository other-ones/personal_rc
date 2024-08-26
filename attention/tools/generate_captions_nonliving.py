import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from datasets_pkgs.mlm_pkgs.consts import BACKGROUNDS,STYLES,NONLIVINGS,NONLIVING_INTERACTIONS
from datasets_pkgs.mlm_pkgs.consts import HUMANS,RELATIVES,NONLIVING_ATTRIBUTES
from datasets_pkgs.mlm_pkgs.consts import ARTISTS,LOCATIONS,COLORS


if __name__=='__main__':
    
    # out=caption_generator.generate_caption()
    # mlm_caption=caption_generator.generate_human_interactions_caption()
    # mlm_caption=caption_generator.generate_object_relations_caption()
    # mlm_caption=caption_generator.generate_styles_caption()
    # mlm_caption=caption_generator.generate_backgrounds_caption()
    # ['human_interactions','object_relations','backgrounds','styles']


    # 1. HUMAN INTERACTIONS
    captions_nonliving_human_interactions=[]
    for interaction in NONLIVING_INTERACTIONS:
        for subject in HUMANS:
            prompt=f"<new1> is {interaction} a {subject}"
            captions_nonliving_human_interactions.append(prompt)
    # 2. object_relations
    captions_nonliving_object_relations=[]
    for rel in RELATIVES:
        for attr in NONLIVING_ATTRIBUTES:
            for other_obj in NONLIVINGS:
                prompt=f"<new1> {rel} a {attr} {other_obj}"
                captions_nonliving_object_relations.append(prompt)

    
    # 3. BACKGROUNDS no ACTION
    captions_nonliving_bg=[]
    for background in BACKGROUNDS:
        prompt1=f"<new1> with the {background} in the background"
        prompt2=f"A view of the <new1> at {background}"
        prompt3=f"<new1> captured at {background}"
        captions_nonliving_bg.append(prompt1)
        captions_nonliving_bg.append(prompt2)
        captions_nonliving_bg.append(prompt3)

    
    # 4. STYLE
    captions_nonliving_style=[]
    for style in STYLES:
        for fmt in ['captured','depicted','rendered',"painted"]:
            prompt1=f"<new1> {fmt} in the {style} style"
            prompt2=f"{style} painting of <new1>"
            captions_nonliving_style.append(prompt1)
            captions_nonliving_style.append(prompt2)
    captions_artist=[]
    for artist in ARTISTS:
        prompt3=f"painting of <new1> by artist {artist}"
        prompt4=f"<new1> painted by artist {artist}"
        captions_artist.append(prompt3)
        captions_artist.append(prompt4)
    captions_nonliving_style+=captions_artist
    
    # 5. ATTRIBUTES
    captions_nonliving_attr=[]
    for attr in NONLIVING_ATTRIBUTES:
        prompt1=f"a {attr} <new1>"
        captions_nonliving_attr.append(prompt1)

    np.random.shuffle(captions_nonliving_human_interactions)
    np.random.shuffle(captions_nonliving_object_relations)
    np.random.shuffle(captions_nonliving_bg)
    np.random.shuffle(captions_nonliving_style)
    np.random.shuffle(captions_nonliving_attr)
    print('INTERACTIONS:',len(captions_nonliving_human_interactions),captions_nonliving_human_interactions[:5])
    print()
    print('OBJECT RELATIONS:',len(captions_nonliving_object_relations),captions_nonliving_object_relations[:5])
    print()
    print('BG:',len(captions_nonliving_bg),captions_nonliving_bg[:5])
    print()
    print('STYLE:',len(captions_nonliving_style),captions_nonliving_style[:5])
    print()
    print('ATTR:',len(captions_nonliving_attr),captions_nonliving_attr[:5])
    print()


    dst_root='../../datasets_pkgs/captions/nonliving'
    os.makedirs(dst_root,exist_ok=True)
    dst_path=os.path.join(dst_root,'captions_nonliving_human_interactions.txt')
    dst_file=open(dst_path,'w')
    captions_nonliving_human_interactions=sorted(captions_nonliving_human_interactions)
    for caption in captions_nonliving_human_interactions:
        dst_file.write("{}\n".format(caption))
    
    dst_path=os.path.join(dst_root,'captions_nonliving_object_relations.txt')
    dst_file=open(dst_path,'w')
    captions_nonliving_object_relations=sorted(captions_nonliving_object_relations)
    for caption in captions_nonliving_object_relations:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_nonliving_bg.txt')
    dst_file=open(dst_path,'w')
    captions_nonliving_bg=sorted(captions_nonliving_bg)
    for caption in captions_nonliving_bg:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_nonliving_style.txt')
    dst_file=open(dst_path,'w')
    captions_nonliving_style=sorted(captions_nonliving_style)
    for caption in captions_nonliving_style:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_nonliving_attr.txt')
    dst_file=open(dst_path,'w')
    captions_nonliving_attr=sorted(captions_nonliving_attr)
    for caption in captions_nonliving_attr:
        dst_file.write("{}\n".format(caption))

