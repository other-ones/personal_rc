import os
from consts_v7 import HUMAN_INTERACTIONS
from consts_v7 import OUTFITS,WEARINGS
from consts_v7 import NONVLIVING_INTERACTIONS_ACTIVE
from consts_v7 import NONLIVINGS,COLORS,HUMANS,RELATIVES,STYLES,BACKGROUNDS
from consts_v7 import NONVLIVING_INTERACTIONS_PASSIVE
from consts_v7 import CREATIVES,SHAPES

def print_variable(variable):
    variable_name = [name for name, value in locals().items() if value is variable][0]
    print(f"Variable name using locals(): {variable_name}")
def namestr(obj, namespace):
    get_name=[name for name in namespace if namespace[name] is obj]
    return get_name[0]
    
ll=[
    HUMAN_INTERACTIONS,
    OUTFITS,
    WEARINGS,
    NONVLIVING_INTERACTIONS_ACTIVE,
    NONLIVINGS,
    COLORS,
    HUMANS,
    RELATIVES,
    STYLES,
    BACKGROUNDS,
    NONVLIVING_INTERACTIONS_PASSIVE,
    NONVLIVING_INTERACTIONS_PASSIVE,
    CREATIVES,
    SHAPES,
]
for item in ll:
    print(namestr(item, globals()),len(set(item)))