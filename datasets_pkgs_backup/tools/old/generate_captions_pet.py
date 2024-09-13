import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from datasets_pkgs.mlm_pkgs.caption_generator_pet import CaptionGeneratorPet
from datasets_pkgs.mlm_pkgs.consts import BACKGROUNDS,STYLES,NONLIVINGS,SOLO_ACTIVITIES,HUMANS,RELATIVES,NONLIVING_ATTRIBUTES
from datasets_pkgs.mlm_pkgs.consts import ARTISTS,LOCATIONS,OUTFITS,COLORS,PET_ATTRIBUTES


if __name__=='__main__':
    caption_generator=CaptionGeneratorPet()
    # out=caption_generator.generate_caption()
    # mlm_caption=caption_generator.generate_human_interactions_caption()
    # mlm_caption=caption_generator.generate_object_relations_caption()
    # mlm_caption=caption_generator.generate_styles_caption()
    # mlm_caption=caption_generator.generate_backgrounds_caption()

    # 1. BACKGROUNDS no ACTION
    captions_pet_bg_noact=[]
    for background in BACKGROUNDS:
        prompt1=f"<new1> with the {background} in the background"
        prompt2=f"<new1> at {background}"
        prompt3=f"<new1> captured at {background}"
        captions_pet_bg_noact.append(prompt1)
        captions_pet_bg_noact.append(prompt2)
        captions_pet_bg_noact.append(prompt3)


    # 2. BACKGROUNDS with ACTION
    captions_pet_bg_act=[]
    for location in LOCATIONS:
        for activity in SOLO_ACTIVITIES:
            prompt3=f"<new1> {activity} {location}"
            captions_pet_bg_act.append(prompt3)
    

    # 3. STYLE
    captions_pet_style=[]
    for style in STYLES:
        for fmt in ['captured','depicted','rendered',"painted"]:
            prompt1=f"<new1> {fmt} in the {style} style"
            prompt2=f"{style} painting of <new1>"
            captions_pet_style.append(prompt1)
            captions_pet_style.append(prompt2)
    captions_artist=[]
    for artist in ARTISTS:
        prompt3=f"painting of <new1> by artist {artist}"
        prompt4=f"<new1> painted by artist {artist}"
        captions_artist.append(prompt3)
        captions_artist.append(prompt4)
    captions_pet_style+=captions_artist


    # 4. RELATIVES
    captions_pet_relations=[]
    for rel in RELATIVES:
        for attr in NONLIVING_ATTRIBUTES:
            for other_obj in NONLIVINGS:
                prompt=f"<new1> {rel} a {attr} {other_obj}"
                captions_pet_relations.append(prompt)


    # 5. HUMAN INTERACTIONS
    captions_pet_interactions=[]
    for act in SOLO_ACTIVITIES:
        for subject in HUMANS:
            if subject[0] in ['a','e','i','o','u']:
                prompt=f"<new1> is {act} with an {subject}"
            elif subject[0].isupper():
                prompt=f"<new1> is {act} with {subject}"
            else:
                prompt=f"<new1> is {act} with a {subject}"
            captions_pet_interactions.append(prompt)
    # 6. OUTFITS
    captions_pet_outfits=[]
    for outfit in OUTFITS:
        for color in COLORS:
            prompt1=f"<new1> wearing {color} {outfit}"
            prompt2=f"<new1> in the {color} {outfit}"
            captions_pet_outfits.append(prompt1)
            captions_pet_outfits.append(prompt2)
    
    # 7. ATTRIBUTES
    captions_pet_attr=[]
    for attr in PET_ATTRIBUTES:
        prompt1=f"a {attr} <new1>"
        captions_pet_attr.append(prompt1)

    np.random.shuffle(captions_pet_bg_act)
    np.random.shuffle(captions_pet_bg_noact)
    np.random.shuffle(captions_pet_style)
    np.random.shuffle(captions_pet_relations)
    np.random.shuffle(captions_pet_interactions)
    np.random.shuffle(captions_pet_attr)
    np.random.shuffle(captions_pet_outfits)
    print('BG NOACT:',len(captions_pet_bg_noact),captions_pet_bg_noact[:5])
    print()
    print('BG ACT:',len(captions_pet_bg_act),captions_pet_bg_act[:5])
    print()
    print('STYLE:',len(captions_pet_style),captions_pet_style[:5])
    print()
    print('RELATIVES:',len(captions_pet_relations),captions_pet_relations[:5])
    print()
    print('INTERACTIONS:',len(captions_pet_interactions),captions_pet_interactions[:5])
    print()
    print('OUTFITS:',len(captions_pet_outfits),captions_pet_outfits[:5])
    print()
    print('ATTR:',len(captions_pet_attr),captions_pet_attr[:5])
    print()

    dst_root='../../datasets_pkgs/captions/pet'
    os.makedirs(dst_root,exist_ok=True)
    dst_path=os.path.join(dst_root,'captions_pet_bg_noact.txt')
    dst_file=open(dst_path,'w')
    captions_pet_bg_noact=sorted(captions_pet_bg_noact)
    for caption in captions_pet_bg_noact:
        dst_file.write("{}\n".format(caption))
    
    dst_path=os.path.join(dst_root,'captions_pet_bg_act.txt')
    dst_file=open(dst_path,'w')
    captions_pet_bg_act=sorted(captions_pet_bg_act)
    for caption in captions_pet_bg_act:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_pet_style.txt')
    dst_file=open(dst_path,'w')
    captions_pet_style=sorted(captions_pet_style)
    for caption in captions_pet_style:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_pet_relations.txt')
    dst_file=open(dst_path,'w')
    captions_pet_relations=sorted(captions_pet_relations)
    for caption in captions_pet_relations:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_pet_interactions.txt')
    dst_file=open(dst_path,'w')
    captions_pet_interactions=sorted(captions_pet_interactions)
    for caption in captions_pet_interactions:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_pet_outfits.txt')
    dst_file=open(dst_path,'w')
    captions_pet_outfits=sorted(captions_pet_outfits)
    for caption in captions_pet_outfits:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_pet_attr.txt')
    dst_file=open(dst_path,'w')
    captions_pet_attr=sorted(captions_pet_attr)
    for caption in captions_pet_attr:
        dst_file.write("{}\n".format(caption))

    # return prompt
    # # styles
# 


