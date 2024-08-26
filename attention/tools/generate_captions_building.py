import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from datasets_pkgs.mlm_pkgs.consts import BACKGROUNDS,STYLES,NONLIVINGS,NONLIVING_INTERACTIONS
from datasets_pkgs.mlm_pkgs.consts import HUMANS,RELATIVES,NONLIVING_ATTRIBUTES
from datasets_pkgs.mlm_pkgs.consts import ARTISTS,LOCATIONS,COLORS,BUILDING_INTERACTIONS,ANIMALS,LIVING_ACTIVITIES,VIEWPOINTS,BUILDING_ATTRIBUTES
from datasets_pkgs.mlm_pkgs.consts import RELATIVES_BUILDING


if __name__=='__main__':
    
    # out=caption_generator.generate_caption()
    # mlm_caption=caption_generator.generate_human_interactions_caption()
    # mlm_caption=caption_generator.generate_object_relations_caption()
    # mlm_caption=caption_generator.generate_styles_caption()
    # mlm_caption=caption_generator.generate_backgrounds_caption()

    # ['human_interactions','object_relations','backgrounds','styles']


    # 1. HUMAN INTERACTIONS
    captions_building_human_interactions=[]
    for interaction in BUILDING_INTERACTIONS:
        for subject in HUMANS:
            prompt=f"<new1> is {interaction} a {subject}"
            captions_building_human_interactions.append(prompt)

    # 2. object_relations
    captions_building_object_relations=[]
    for rel in RELATIVES_BUILDING:
        for attr in NONLIVING_ATTRIBUTES:
            for other_obj in NONLIVINGS:
                prompt=f"a {attr} {other_obj} {rel} <new1>"
                captions_building_object_relations.append(prompt)

    
    # 3. BACKGROUNDS
    captions_building_bg=[]
    for background in BACKGROUNDS:
        prompt1=f"<new1> with the {background} in the background"
        prompt2=f"A view of the <new1> at {background}"
        prompt3=f"<new1> captured at {background}"
        captions_building_bg.append(prompt1)
        captions_building_bg.append(prompt2)
        captions_building_bg.append(prompt3)
    
    # 2. BACKGROUNDS with ACTION
    captions_building_animal_act=[]
    for animal in ANIMALS:
        for activity in LIVING_ACTIVITIES:
            prompt3=f"{animal} {activity} in front of <new1>"
            captions_building_animal_act.append(prompt3)

    

    

    # 4. STYLE
    captions_building_style=[]
    for style in STYLES:
        for fmt in ['captured','depicted','rendered',"painted"]:
            prompt1=f"<new1> {fmt} in the {style} style"
            prompt2=f"{style} painting of <new1>"
            captions_building_style.append(prompt1)
            captions_building_style.append(prompt2)
    captions_artist=[]
    for artist in ARTISTS:
        prompt3=f"painting of <new1> by artist {artist}"
        prompt4=f"<new1> painted by artist {artist}"
        captions_artist.append(prompt3)
        captions_artist.append(prompt4)
    captions_building_style+=captions_artist


    
    
    # 5. ATTRIBUTES
    captions_building_attr=[]
    for attr in BUILDING_ATTRIBUTES:
        prompt1=f"a {attr} <new1>"
        captions_building_attr.append(prompt1)

    # 6. viewpoints
    captions_building_view=[]
    for view in VIEWPOINTS:
        prompt1=f"{view} <new1>"
        captions_building_view.append(prompt1)
    
    
    np.random.shuffle(captions_building_human_interactions)
    np.random.shuffle(captions_building_object_relations)
    np.random.shuffle(captions_building_bg)
    np.random.shuffle(captions_building_style)
    np.random.shuffle(captions_building_attr)
    np.random.shuffle(captions_building_animal_act)
    np.random.shuffle(captions_building_view)
    print('INTERACTIONS:',len(captions_building_human_interactions),captions_building_human_interactions[:5])
    print()
    print('RELATIVES:',len(captions_building_object_relations),captions_building_object_relations[:5])
    print()
    print('BG:',len(captions_building_bg),captions_building_bg[:5])
    print()
    print('STYLE:',len(captions_building_style),captions_building_style[:5])
    print()
    print('ATTR:',len(captions_building_attr),captions_building_attr[:5])
    print()
    print('ANIMAL ACT:',len(captions_building_animal_act),captions_building_animal_act[:5])
    print()
    print('VIEW:',len(captions_building_view),captions_building_view[:5])
    print()

    dst_root='../../datasets_pkgs/captions/building'
    os.makedirs(dst_root,exist_ok=True)
    dst_path=os.path.join(dst_root,'captions_building_human_interactions.txt')
    dst_file=open(dst_path,'w')
    captions_building_human_interactions=sorted(captions_building_human_interactions)
    for caption in captions_building_human_interactions:
        dst_file.write("{}\n".format(caption))
    
    dst_path=os.path.join(dst_root,'captions_building_object_relations.txt')
    dst_file=open(dst_path,'w')
    captions_building_object_relations=sorted(captions_building_object_relations)
    for caption in captions_building_object_relations:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_building_bg.txt')
    dst_file=open(dst_path,'w')
    captions_building_bg=sorted(captions_building_bg)
    for caption in captions_building_bg:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_building_style.txt')
    dst_file=open(dst_path,'w')
    captions_building_style=sorted(captions_building_style)
    for caption in captions_building_style:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_building_attr.txt')
    dst_file=open(dst_path,'w')
    captions_building_attr=sorted(captions_building_attr)
    for caption in captions_building_attr:
        dst_file.write("{}\n".format(caption))


    dst_path=os.path.join(dst_root,'captions_building_animal_act.txt')
    dst_file=open(dst_path,'w')
    captions_building_animal_act=sorted(captions_building_animal_act)
    for caption in captions_building_animal_act:
        dst_file.write("{}\n".format(caption))


    dst_path=os.path.join(dst_root,'captions_building_view.txt')
    dst_file=open(dst_path,'w')
    captions_building_view=sorted(captions_building_view)
    for caption in captions_building_view:
        dst_file.write("{}\n".format(caption))

    # # styles
# 


