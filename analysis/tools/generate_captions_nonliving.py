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
    captions_nonliving_interactions=[]
    for interaction in NONLIVING_INTERACTIONS:
        for subject in HUMANS:
            prompt=f"<new> is {interaction} a {subject}"
            captions_nonliving_interactions.append(prompt)
    # 2. object_relations
    captions_nonliving_relations=[]
    for rel in RELATIVES:
        for attr in NONLIVING_ATTRIBUTES:
            for other_obj in NONLIVINGS:
                prompt=f"<new> {rel} a {attr} {other_obj}"
                captions_nonliving_relations.append(prompt)

    
    # 3. BACKGROUNDS no ACTION
    captions_nonliving_bg=[]
    for background in BACKGROUNDS:
        prompt1=f"<new> with the {background} in the background"
        prompt2=f"A view of the <new> at {background}"
        prompt3=f"<new> captured at {background}"
        captions_nonliving_bg.append(prompt1)
        captions_nonliving_bg.append(prompt2)
        captions_nonliving_bg.append(prompt3)

    

    

    # 4. STYLE
    captions_nonliving_style=[]
    for style in STYLES:
        for fmt in ['captured','depicted','rendered',"painted"]:
            prompt1=f"<new> {fmt} in the {style} style"
            prompt2=f"{style} painting of <new>"
            captions_nonliving_style.append(prompt1)
            captions_nonliving_style.append(prompt2)
    captions_artist=[]
    for artist in ARTISTS:
        prompt3=f"painting of <new1> by artist f{artist}"
        prompt4=f"<new1> painted by artist f{artist}"
        captions_artist.append(prompt3)
        captions_artist.append(prompt4)
    captions_nonliving_style+=captions_artist


    
    
    # 5. ATTRIBUTES
    captions_nonliving_attr=[]
    for attr in NONLIVING_ATTRIBUTES:
        prompt1=f"a {attr} <new>"
        captions_nonliving_attr.append(prompt1)

    np.random.shuffle(captions_nonliving_interactions)
    np.random.shuffle(captions_nonliving_relations)
    np.random.shuffle(captions_nonliving_style)
    np.random.shuffle(captions_nonliving_attr)
    print('INTERACTIONS:',len(captions_nonliving_interactions),captions_nonliving_interactions[:5])
    print('RELATIVES:',len(captions_nonliving_relations),captions_nonliving_relations[:5])
    print('BG:',len(captions_nonliving_bg),captions_nonliving_bg[:5])
    print('STYLE:',len(captions_nonliving_style),captions_nonliving_style[:5])
    print('ATTR:',len(captions_nonliving_attr),captions_nonliving_attr[:5])


    dst_root='../../datasets_pkgs/captions'
    os.makedirs(dst_root,exist_ok=True)
    dst_path=os.path.join(dst_root,'captions_nonliving_interactions.txt')
    dst_file=open(dst_path,'w')
    captions_nonliving_interactions=sorted(captions_nonliving_interactions)
    for caption in captions_nonliving_interactions:
        dst_file.write("{}\n".format(caption))
    
    dst_path=os.path.join(dst_root,'captions_nonliving_relations.txt')
    dst_file=open(dst_path,'w')
    captions_nonliving_relations=sorted(captions_nonliving_relations)
    for caption in captions_nonliving_relations:
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

    dst_path=os.path.join(dst_root,'captions_nonliving_attr.txt')
    dst_file=open(dst_path,'w')
    captions_nonliving_attr=sorted(captions_nonliving_attr)
    for caption in captions_nonliving_attr:
        dst_file.write("{}\n".format(caption))

    # return prompt
    # # styles
# 


