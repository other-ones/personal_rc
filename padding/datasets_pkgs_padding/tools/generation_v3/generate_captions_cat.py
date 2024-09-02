import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from datasets_pkgs.mlm_pkgs.consts import BACKGROUNDS,STYLES,NONLIVINGS,SOLO_ACTIVITIES,HUMANS,RELATIVES,NONLIVING_ATTRIBUTES
from datasets_pkgs.mlm_pkgs.consts import ARTISTS,LOCATIONS,IN_OUTFITS,WEARINGS,COLORS,CAT_ATTRIBUTES,CREATIONS
from datasets_pkgs.mlm_pkgs.consts import CREATION_ATTRIBUTES
import shutil

if __name__=='__main__':
    dst_root='../../datasets_pkgs/captions/cat'
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
        print(dst_root,'deleted')
    os.makedirs(dst_root,exist_ok=True)

    # 1. BACKGROUNDS no ACTION
    captions_cat_bg_noact=[]
    for background in BACKGROUNDS:
        prompt=f"<new1> with the {background} in the background"
        prompt=f"<new1> captured at {background}"
        captions_cat_bg_noact.append(prompt)
        captions_cat_bg_noact.append(prompt)
    for location in LOCATIONS:
        prompt=f"<new1> {location}"
        captions_cat_bg_noact.append(prompt)


    # 2. BACKGROUNDS with ACTION
    captions_cat_bg_act=[]
    for location in LOCATIONS:
        for activity in SOLO_ACTIVITIES:
            prompt=f"<new1> {activity} {location}"
            captions_cat_bg_act.append(prompt)
    

    # 3. STYLE
    captions_cat_style=[]
    for style in STYLES:
        for fmt in ['captured','depicted','rendered',"painted"]:
            prompt=f"<new1> {fmt} in the {style} style"
            captions_cat_style.append(prompt)
            prompt=f"{style} painting of <new1>"
            captions_cat_style.append(prompt)
    captions_artist=[]
    for artist in ARTISTS:
        prompt=f"<new1> painting by artist {artist}"
        captions_artist.append(prompt)
        prompt=f"<new1> painted by artist {artist}"
        captions_artist.append(prompt)
        # captions_artist.append(prompt3)
    captions_cat_style+=captions_artist


    # 4. RELATIVES
    captions_cat_relations=[]
    for rel in RELATIVES:
        for attr in NONLIVING_ATTRIBUTES:
            for other_obj in NONLIVINGS:
                prompt=f"<new1> {rel} a {attr} {other_obj}"
                captions_cat_relations.append(prompt)


    # # 5. HUMAN INTERACTIONS
    captions_cat_human_interactions=[]
    for act in SOLO_ACTIVITIES:
        for subject in HUMANS:
            if subject[0] in ['a','e','i','o','u']:
                prompt=f"<new1> is {act} with an {subject}"
            elif subject[0].isupper():
                prompt=f"<new1> is {act} with {subject}"
            else:
                prompt=f"<new1> is {act} with a {subject}"
            captions_cat_human_interactions.append(prompt)


    # 6. OUTFITS
    captions_cat_outfits=[]
    for outfit in IN_OUTFITS:
        for color in COLORS:
            prompt=f"<new1> in {color} {outfit}"
            captions_cat_outfits.append(prompt)
    for wearing in WEARINGS:
        for color in COLORS:
            prompt=f"<new1> wearing {color} {wearing}"
            captions_cat_outfits.append(prompt)
    captions_cat_act_outfits=[]
    for outfit in IN_OUTFITS:
        for act in SOLO_ACTIVITIES:
            prompt=f"<new1> in {outfit} is {act}"
            captions_cat_act_outfits.append(prompt)

    captions_cat_bg_outfits=[]
    for outfit in IN_OUTFITS:
        for location in LOCATIONS:
            prompt=f"<new1> in {outfit} {location}"
            captions_cat_bg_outfits.append(prompt)
    
    # 7. ATTRIBUTES
    captions_cat_attr=[]
    for attr in CAT_ATTRIBUTES:
        prompt=f"{attr} <new1>"
        captions_cat_attr.append(prompt)
    for attr in CAT_ATTRIBUTES:
        for location in LOCATIONS:
            prompt=f"{attr} <new1> {location}"
            captions_cat_attr.append(prompt)

    # 8. Creations
    captions_cat_creations=[]
    for creation in CREATIONS:
        for attr in CREATION_ATTRIBUTES:
            prompt=f"{attr} {creation} <new1>"
            captions_cat_creations.append(prompt)
    

    np.random.shuffle(captions_cat_bg_act)
    np.random.shuffle(captions_cat_bg_noact)
    np.random.shuffle(captions_cat_style)
    np.random.shuffle(captions_cat_relations)
    np.random.shuffle(captions_cat_human_interactions)
    np.random.shuffle(captions_cat_attr)
    np.random.shuffle(captions_cat_outfits)
    np.random.shuffle(captions_cat_act_outfits)
    np.random.shuffle(captions_cat_creations)
    np.random.shuffle(captions_cat_bg_outfits)
    print('BG NOACT:',len(captions_cat_bg_noact))
    for item in captions_cat_bg_noact[:5]:
        print(item)
    print()
    print('BG ACT:',len(captions_cat_bg_act))
    for item in captions_cat_bg_act[:5]:
        print(item)
    # print()
    print('STYLE:',len(captions_cat_style))
    for item in captions_cat_style[:5]:
        print(item)
    print()

    print('RELATIVES:',len(captions_cat_relations))
    for item in captions_cat_relations[:5]:
        print(item)
    print()

    print('INTERACTIONS:',len(captions_cat_human_interactions))
    for item in captions_cat_human_interactions[:5]:
        print(item)
    print()


    print('OUTFITS:',len(captions_cat_outfits))
    for item in captions_cat_outfits[:5]:
        print(item)
    print()

    print('ACT+OUTFITS:',len(captions_cat_act_outfits))
    for item in captions_cat_act_outfits[:5]:
        print(item)
    print()

    print('BG+OUTFITS:',len(captions_cat_bg_outfits))
    for item in captions_cat_bg_outfits[:5]:
        print(item)
    print()

    print('ATTR:',len(captions_cat_attr))
    for item in captions_cat_attr[:5]:
        print(item)
    print()

    print('CREATIONS:',len(captions_cat_creations))
    for item in captions_cat_creations[:5]:
        print(item)
    print()

    
    dst_path=os.path.join(dst_root,'captions_cat_bg_noact.txt')
    dst_file=open(dst_path,'w')
    captions_cat_bg_noact=list(set(captions_cat_bg_noact))
    captions_cat_bg_noact=sorted(captions_cat_bg_noact)
    for caption in captions_cat_bg_noact:
        dst_file.write("{}\n".format(caption))
    
    dst_path=os.path.join(dst_root,'captions_cat_bg_act.txt')
    dst_file=open(dst_path,'w')
    captions_cat_bg_act=list(set(captions_cat_bg_act))
    captions_cat_bg_act=sorted(captions_cat_bg_act)
    for caption in captions_cat_bg_act:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_cat_style.txt')
    dst_file=open(dst_path,'w')
    captions_cat_style=list(set(captions_cat_style))
    captions_cat_style=sorted(captions_cat_style)
    for caption in captions_cat_style:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_cat_relations.txt')
    dst_file=open(dst_path,'w')
    captions_cat_relations=list(set(captions_cat_relations))
    captions_cat_relations=sorted(captions_cat_relations)
    for caption in captions_cat_relations:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_cat_human_interactions.txt')
    dst_file=open(dst_path,'w')
    captions_cat_human_interactions=list(set(captions_cat_human_interactions))
    captions_cat_human_interactions=sorted(captions_cat_human_interactions)
    for caption in captions_cat_human_interactions:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_cat_outfits.txt')
    dst_file=open(dst_path,'w')
    captions_cat_outfits=list(set(captions_cat_outfits))
    captions_cat_outfits=sorted(captions_cat_outfits)
    for caption in captions_cat_outfits:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_cat_act_outfits.txt')
    dst_file=open(dst_path,'w')
    captions_cat_act_outfits=list(set(captions_cat_act_outfits))
    captions_cat_act_outfits=sorted(captions_cat_act_outfits)
    for caption in captions_cat_act_outfits:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_cat_attr.txt')
    dst_file=open(dst_path,'w')
    captions_cat_attr=list(set(captions_cat_attr))
    captions_cat_attr=sorted(captions_cat_attr)
    for caption in captions_cat_attr:
        dst_file.write("{}\n".format(caption))


    dst_path=os.path.join(dst_root,'captions_cat_creations.txt')
    dst_file=open(dst_path,'w')
    captions_cat_creations=list(set(captions_cat_creations))
    captions_cat_creations=[item.strip() for item in captions_cat_creations if item.strip()]
    captions_cat_creations=sorted(captions_cat_creations)
    for caption in captions_cat_creations:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_cat_bg_outfits.txt')
    dst_file=open(dst_path,'w')
    captions_cat_bg_outfits=list(set(captions_cat_bg_outfits))
    captions_cat_bg_outfits=[item.strip() for item in captions_cat_bg_outfits if item.strip()]
    captions_cat_bg_outfits=sorted(captions_cat_bg_outfits)
    for caption in captions_cat_bg_outfits:
        dst_file.write("{}\n".format(caption))

    dst_path=os.path.join(dst_root,'captions_cat_specific.txt')
    dst_file=open(dst_path,'w')
    captions_cat_specific=open("../../datasets_pkgs/captions/specific_captions/cat_specific_captions.txt").readlines()
    captions_cat_specific=list(set(captions_cat_specific))
    captions_cat_specific=[item.strip() for item in captions_cat_specific if item.strip()]
    captions_cat_specific=sorted(captions_cat_specific)
    for caption in captions_cat_specific:
        dst_file.write("{}\n".format(caption))



