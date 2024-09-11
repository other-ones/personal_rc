import numpy as np
import shutil
import os
from consts import HUMANS,SOLO_ACTIVITIES,BACKGROUNDS,LOCATIONS,ANIMALS
from consts import HUMAN_INTERACTIONS
from consts import NONLIVING_INTERACTIONS,NONLIVING_ATTRIBUTES,COLORS,RELATIVES
from consts import STYLES,WEARINGS,ARTISTS,NONLIVINGS,OUTFITS
from consts import NONVLIVING_INTERACTIONS_PASSIVE,NONVLIVING_INTERACTIONS_ACTIVE

def main():
    dst_root='../../captions/contextnet/living'
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
        print(dst_root,'deleted')
    os.makedirs(dst_root,exist_ok=True)




    # 1. SOLO activity
    captions_solo_act=[]
    for solo_act in SOLO_ACTIVITIES:
        for subject in (HUMANS+ANIMALS):
            # for location in LOCATIONS:
            if subject[0] in ['a','e','i','o','u']:
                prompt=f"an {subject} {solo_act}"
            elif subject[0].isupper():
                prompt=f"{subject} {solo_act}"
            else:
                prompt=f"a {subject} {solo_act}"
            captions_solo_act.append(prompt)
    captions_solo_act=list(set(captions_solo_act))
    np.random.shuffle(captions_solo_act)
    captions_solo_act=captions_solo_act[:int(1e5)]
    dst_file=open(os.path.join(dst_root,'captions_solo_act.txt'),'w')
    for item in sorted(captions_solo_act):
        dst_file.write("{}\n".format(item))
    samples=np.random.choice(captions_solo_act,5)
    print('\n')
    print(f'SOLO_ACT {len(captions_solo_act)}')
    for item in samples:
        print(item)





    # 2. BACKGROUNDS
    captions_backgrounds=[]
    for subject in (HUMANS+ANIMALS+NONLIVINGS):
        for background in BACKGROUNDS:
            prompt=f"{subject} with the {background} in the background"
            captions_backgrounds.append(prompt)
            prompt=f"{subject} viewed with the {background}"
            captions_backgrounds.append(prompt)
            # # prompt=f"{subject} captured with the {background} in the background"
            # # captions_backgrounds.append(prompt)
            # # prompt=f"{subject} viewed with the {background}"
            # captions_backgrounds.append(prompt)
    captions_backgrounds=list(set(captions_backgrounds))
    np.random.shuffle(captions_backgrounds)
    captions_backgrounds=captions_backgrounds[:int(1e5)]
    dst_file=open(os.path.join(dst_root,'captions_backgrounds.txt'),'w')
    for item in sorted(captions_backgrounds):
        dst_file.write("{}\n".format(item))
    samples=np.random.choice(captions_backgrounds,5)
    print('\n')
    print(f'BACKGROUNDS {len(captions_backgrounds)}')
    for item in samples:
        print(item)





    # 3. RELATIVE POSITIONS
    captions_relations=[]
    for rel in RELATIVES:
        for color in COLORS:
            for subject in (HUMANS+NONLIVINGS):
                if subject[0] in ['a','e','i','o','u']:
                    subject_word=f"an {subject}"
                elif subject[0].isupper():
                    subject_word=f"{subject}"
                else:
                    subject_word=f"a {subject}" 
                
                for other_obj in NONLIVINGS:
                    prompt=f"{subject_word} {rel} a {color} {other_obj}"
                    captions_relations.append(prompt)
    captions_relations=list(set(captions_relations))
    np.random.shuffle(captions_relations)
    captions_relations=captions_relations[:int(1e5)]
    dst_file=open(os.path.join(dst_root,'captions_relations.txt'),'w')
    for item in sorted(captions_relations):
        dst_file.write("{}\n".format(item))
    samples=np.random.choice(captions_relations,5)
    print('\n')
    print('POSITION')
    print(f'POSITION {len(captions_relations)}')

    for item in samples:
        print(item)

    
    # 4. STYLE
    captions_styles=[]
    for subject in (ANIMALS+HUMANS+NONLIVINGS):
        for fmt in ['captured','depicted','rendered','painted']:
            for style in STYLES:
                prompt=f"{subject} {fmt} in the {style} style"
                captions_styles.append(prompt)
    for subject in (ANIMALS+HUMANS):
        for artist in ARTISTS:
            prompt=f"{subject} painted by {artist}"
            captions_styles.append(prompt)
    captions_styles=list(set(captions_styles))
    np.random.shuffle(captions_styles)
    captions_styles=captions_styles[:int(1e5)]
    dst_file=open(os.path.join(dst_root,'captions_styles.txt'),'w')
    for item in sorted(captions_styles):
        dst_file.write("{}\n".format(item))
    samples=np.random.choice(captions_styles,5)
    print('\n')
    print(f'STYLE {len(captions_styles)}')
    for item in samples:
        print(item)

    # 5. WEARING
    captions_wearings=[]
    for subject in (ANIMALS+HUMANS):
        for outfit in OUTFITS:
            for color in COLORS:
                prompt=f"{subject} in a {color} {outfit}"
                captions_wearings.append(prompt)
                prompt=f"{subject} dressed in a {color} {outfit}"
                captions_wearings.append(prompt)
    captions_wearings=list(set(captions_wearings))
    np.random.shuffle(captions_wearings)
    captions_wearings=captions_wearings[:int(1e5)]
    dst_file=open(os.path.join(dst_root,'captions_wearings.txt'),'w')
    for item in sorted(captions_wearings):
        dst_file.write("{}\n".format(item))
    samples=np.random.choice(captions_wearings,5)
    print('\n')
    print(f'WEARING {len(captions_wearings)}')
    for item in samples:
        print(item)


    # 6. NONLIVING INTERACT
    captions_interact=[]
    for subject in HUMANS:
        for interact in NONVLIVING_INTERACTIONS_PASSIVE:
            for nonliving in (NONLIVINGS):
                # for location in LOCATIONS:
                if nonliving[0] in ['a','e','i','o','u']:
                    prompt=f"an {nonliving} is {interact} by {subject}"
                else:
                    prompt=f"a {nonliving} is {interact} by {subject}"
                captions_interact.append(prompt)
        for interact in NONVLIVING_INTERACTIONS_ACTIVE:
            for nonliving in (NONLIVINGS):
                # parse noliving_object
                if nonliving[0] in ['a','e','i','o','u']:
                    nonliving_word=f"an {nonliving}"
                else:
                    nonliving_word=f"a {nonliving}"
                # parse subject
                if subject[0] in ['a','e','i','o','u']:
                    subject_word=f"an {subject}"
                elif subject[0].isupper():
                    subject_word=f"{subject}"
                else:
                    subject_word=f"a {subject}" 
                prompt=f"{subject_word} is {interact} {nonliving_word}"
                captions_interact.append(prompt)
    captions_interact=list(set(captions_interact))
    np.random.shuffle(captions_interact)
    captions_interact=captions_interact[:int(1e5)]
    dst_file=open(os.path.join(dst_root,'captions_interact.txt'),'w')
    for item in sorted(captions_interact):
        dst_file.write("{}\n".format(item))
    samples=np.random.choice(captions_interact,5)
    print('\n')
    print(f'INTERACT {len(captions_interact)}')
    for item in samples:
        print(item)

if __name__=='__main__':
    main()