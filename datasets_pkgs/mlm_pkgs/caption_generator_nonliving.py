import numpy as np
from .caption_generator import CaptionGenerator
from consts import NONLIVING_ATTRIBUTES,NONLIVING_INTERACTIONS,HUMANS,BACKGROUNDS,NONLIVINGS,RELATIVES,STYLES
from consts import ARTISTS,LOCATIONS
interactions = [
    'inspected by', 'discussed by', 'examined by',
    'repaired by', 'sketched by', 'cleaned by', 'photographed by',
    'packaged by', 'analyzed by', 'painted by', 'critiqued by',
    'used in a scene by', 'performed by','filmed by', 'photographed by', 'transported by',
    'restored by a', 'designed by', 'evaluated by',
    'auctioned by an', 'played on by', 'modeled by',
    'recited by', 'implemented by','researched by',
    'presented by', 'juggled by', 'planted by'
]


class CaptionGeneratorNonLiving(CaptionGenerator):
    def __init__(self):
        super().__init__()
        self.types=['human_interactions','object_relations','backgrounds','styles']
    def generate_caption(self):
        sampled_type=np.random.choice(self.types)
        if sampled_type=='human_interactions':
            mlm_caption=self.generate_human_interactions_caption()
        elif sampled_type=='object_relations':
            mlm_caption=self.generate_object_relations_caption()
        elif sampled_type=='backgrounds':
            mlm_caption=self.generate_backgrounds_caption()
        elif sampled_type=='styles':
            mlm_caption=self.generate_styles_caption() 
        else:
            assert False
        return mlm_caption
    
    def generate_triplet(self):
        sampled_type_pos,sampled_type_neg=np.random.choice(self.types,size=2)
        if sampled_type_pos=='human_interactions':
            anchor=self.generate_human_interactions_caption()
        elif sampled_type_pos=='object_relations':
            anchor=self.generate_object_relations_caption()
        elif sampled_type_pos=='backgrounds':
            anchor=self.generate_backgrounds_caption()
        elif sampled_type_pos=='styles':
            anchor=self.generate_styles_caption() 

        if sampled_type_neg=='human_interactions':
            neg=self.generate_human_interactions_caption()
        elif sampled_type_neg=='object_relations':
            neg=self.generate_object_relations_caption()
        elif sampled_type_neg=='backgrounds':
            neg=self.generate_backgrounds_caption()
        elif sampled_type_neg=='styles':
            neg=self.generate_styles_caption() 

        return anchor,neg
    def generate_human_interactions_caption(self):
        interaction=np.random.choice(NONLIVING_INTERACTIONS)
        subject=np.random.choice(HUMANS)
        prompt=f"<new> is {interaction} a {subject}"
        return prompt
    
    def generate_object_relations_caption(self):
        rel=np.random.choice(RELATIVES)        
        attr=np.random.choice(NONLIVING_ATTRIBUTES)        
        other_obj=np.random.choice(NONLIVINGS)  
        if np.random.rand()<0.5:      
            prompt=f"<new> {rel} a {attr} {other_obj}"
        else:
            location=np.random.choice(LOCATIONS)
            prompt=f"<new> {rel} a {attr} {other_obj} {location}"
        return prompt
    
    def generate_backgrounds_caption(self):
        background=np.random.choice(BACKGROUNDS)
        prompt=f"<new> with the {background} in the background"
        return prompt
    
    def generate_styles_caption(self):
        fmt=np.random.choice(['captured','depicted','rendered',"painted"])
        style=np.random.choice(STYLES)
        choice=["type1","type2","type3"]
        if choice=='type1':
            prompt=f"<new> {fmt} in the {style} style"
        elif choice=="type2":
            prompt=f"{style} painting of <new>"
        else:
            artist=np.random.choice(ARTISTS)
            prompt=f"painting of <new1> by artist f{artist}"
        return prompt