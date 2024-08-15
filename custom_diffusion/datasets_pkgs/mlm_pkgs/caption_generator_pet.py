import numpy as np
from .caption_generator import CaptionGenerator


subjects = [
    "teacher", "a soldier", "magician", "acrobat", "explorer", "scientist", "artist", "writer", "detective", "athlete",
    "nurse", "doctor", "politician", "monk", "priest", "sailor", "pilot", "chef", "farmer", "hunter", "fisherman",
    "trader", "smith", "jeweler", "tailor", "gardener", "librarian", "architect", "engineer", "programmer", "astronaut",
    "alien", "vampire", "wizard", "knight", "samurai", "ninja", "king", "queen", "prince", "princess", "bard", "mercenary",
    "cyborg", "robot", "android", "superhero", "villain", "Albert Einstein", "Marie Curie", "Leonardo da Vinci",
    "Cleopatra", "William Shakespeare", "Mahatma Gandhi", "Winston Churchill", "Amelia Earhart", "Abraham Lincoln",
    "Martin Luther King Jr.", "Frida Kahlo", "Pablo Picasso",  "Stephen Hawking",
    "Beethoven", "Mozart", "Vincent van Gogh","Isaac Newton", "Galileo Galilei",
    "Marco Polo", "Socrates", "Aristotle", "Tutankhamun", "Queen Elizabeth I", "Genghis Khan",
]

interactions = [
    "playing with", "talking to", "feeding", "hugging", "walking", "training", "petting", 
    "brushing", "cuddling with", "carrying", "photographing", "watching", "dressing up", 
    "giving a bath to", "cleaning up after", "taking to the vet", "grooming", "teaching tricks to", 
    "running with", "sleeping beside", "reading to", "celebrating a birthday with", 
    "sharing food with", "taking a selfie with", "protecting", "napping with", "taking for a car ride", 
    "playing fetch with", "riding in a bicycle basket with", "playing in the park with", "resting beside", 
    "sitting on a bench with", "enjoying a sunny day with", "sharing a secret with", 
    "making a video with", "having a picnic with", "watching TV with", "doing yoga with", 
    "playing a game with", "taking a walk in the rain with", "going for a swim with", 
    "enjoying a sunset with", "having breakfast with", "dancing with", "skateboarding with", 
    "hiking with", "watching a movie with", "going on an adventure with"
]

backgrounds = [
    "Great Wall of China", "Times Square", "botanic garden",
    "Sahara Desert", "coral reef", "Mount Everest base",
    "Amazon rainforest", "Venetian canal", "Paris cafe",
    "modern art museum", "Silicon Valley tech lab", "Egyptian pyramid",
    "Hollywood movie set", "medieval castle", "Tokyo skyline",
    "Antarctic ice field", "Caribbean beach", "international space station",
    "fairy tale castle", "major city public library", "Renaissance chapel",
    "Roman Colosseum", "Grand Canyon", "Eiffel Tower",
    "abandoned amusement park", "Old West ghost town", "lunar base",
    "Alaskan tundra", "mega shopping mall", "remote mountain valley",
    "Las Vegas Strip", "Australian Outback", "North Pole",
    "luxury Mediterranean yacht", "rural farm", "lively street market",
    "ancient Greek agora", "space exhibition", "crowded metro station",
    "Scandinavian minimalist kitchen", "nostalgic old bookstore", "grand opera house",
    "mythical underwater Atlantis", "cutting-edge Silicon Valley startup", "secluded Himalayan monastery",
    "quiet suburban neighborhood", "towering downtown skyscraper", "ancient Egyptian tomb",
    "sunset beach", "tranquil rooftop garden", "busy factory floor",
    "elegant luxury hotel lobby", "bustling college campus", "seaside boardwalk",
    "lush vineyard", "craft brewery", "sleek modern office",
    "packed football stadium", "tranquil desert oasis", "central train station",
    "spacious old warehouse", "urban chic rooftop", "crumbling ancient ruins",
    "steep mountain pass", "serene city park", "thick deep forest",
    "electric rock concert", "industrial manufacturing zone", "refined art gallery",
    "historic city bridge", "vibrant digital world", "futuristic neon-lit city",
    "submarine under the Arctic", "large airplane hangar", "remote arctic village",
    "mystical Baltic sea coast", "bustling Middle Eastern bazaar", "serene Nordic fjord",
    "idyllic Pacific island", "vast African savannah", "ancient Neolithic henge",
    "mysterious underground cave", "medieval European market", "opulent Oriental palace",
    "exotic South American jungle", "foreboding deep sea trench", "iconic Hollywood Boulevard",
    "majestic European cathedral", "legendary Wild West saloon", "utopian space colony",
    "Victorian Gothic manor", "swashbuckling pirate ship", "World War I trench",
    "nomadic Mongolian steppe", "secluded Amazonian lodge", "elegant Russian ballet stage",
    "soulful New Orleans jazz club", "natural Appalachian trail", "vibrant Cuban cigar lounge",
    "eclectic antique shop", "top of a modern skyscraper", "bustling International Space Station",
    "wild Scottish highland", "active volcanic crater", "royal court of Versailles",
    "time-worn ancient Greek temple", "colorful Indian spice market", "cutting-edge Arctic research station",
    "high-powered corporate office", "dungeon beneath medieval castle", "endless Siberian railway",
    "sun-drenched California vineyard", "high-speed car racing track", "stark Mexican desert",
    "mysterious Gothic castle", "lively Venice Beach boardwalk", "quaint French vineyard",
    "secluded Zen garden", "open-air Moroccan market", "industrial Detroit factory",
    "space-age Tokyo district", "old European cobblestone street", "modern Singapore skyline",
    "Isolated Icelandic hot spring", "traditional Maori village", "Middle Ages Scottish castle",
    "energetic Brazilian carnival", "tranquil Swedish forest", "traditional Turkish bath",
    "innovative Berlin tech hub", "picturesque Swiss mountain town", "majestic Mount Olympus",
    "isolated desert highway", "opulent Monaco casino", "historic Pilgrim landing site",
    "San Francisco cable car", "enigmatic Stonehenge", "vibrant Rio de Janeiro carnival"
]
other_objs = [
    "microscope", "spaceship", "magic wand", "laptop", "drone", "painting", "musical score", "ancient manuscript",
    "map", "motorcycle", "telescope", "camera", "bicycle", "paintbrush", "musical instrument",
    "garden tools", "smartphone", "book", "screwdriver", "helmet", "chessboard", "surfboard", "globe",
    "stopwatch",  "suitcase", "electric guitar", "basketball", "yoga mat",
    "chandelier", "sculpture", "typewriter", "sewing machine", "binoculars", "skateboard", "flashlight",
    "pillow", "bracelet", "watch", "skis", "snowboard", "fishing rod", "bow and arrow", "model train",
    "puzzle", "stuffed animal",  "dice", "candle", "fan", "boots", "sandal",
    "hat", "scarf", "belt", "necklace", "pearls", "perfume bottle", "makeup brush", "eyeglasses",
    "wristband", "keychain", "wallet", "pen", "pencil", "notebook", "stamp", "coin", "vinyl record","sofa"
]
colors = [
    "red", "blue", "green", "yellow", "black", "white",
    "orange", "purple", "gray", "brown", "silver", "gold",
    "beige", "ivory", "teal", "navy", "maroon", "turquoise",
    "lime", "charcoal", "coral", "cyan", "magenta", "olive",
    "chocolate", "mustard","pink","amber","burgundy","copper","mint",
    "Violet","indigo","crimson","ruby","emerald","slate","sapphire",
]
colors=list(set(colors))
relative_words = [
    "beside", "near", "next to", "behind", "in front of",
    "above", "below", "across from", "alongside", "amid",
    "among", "atop", "against", "within", "surrounding",
    "beneath", "over", "under", "to the left of", "to the right of",
    "on top of", "hanging over", "leaning on", "framed by",
    "encircled by", "interspersed with", "between", "flanked by",
    "opposite", "aligned with", "adjacent to", "tucked behind",
    "sprawled in front of", "nestled among", "mounted on",
    "clinging to", "perched on", "suspended above", "arrayed around",
    "bordering"
]
styles = [
    "Art Nouveau", "Art Deco", "Abstract", "Baroque", "Bauhaus", "Byzantine", "Cubism", "Dada", "Expressionism", "Fauvism",
    "Futurism", "Gothic", "Graffiti", "Impressionism", "Minimalism", "Modernism", "Neo-Impressionism", "Neoclassicism",
    "Op Art", "Pop Art", "Post-Impressionism", "Renaissance", "Rococo", "Surrealism", "Symbolism",
    "Acrylic", "Charcoal", "Digital", "Engraving", "Ink", "Oil", "Pastel", "Pencil", "Screenprint",
    "Stencil", "Tempera", "Watercolor", "Woodcut", "Anime", "Cartoon", "Claymation", "Collage", "Comic", "Concept Art",
    "Pixel Art", "Silhouette", "Sketch", "Stencil Art", "Street Art", "Textile Art", "Vector Art", "Vexel", "Video Art",
    "Vintage", "3D Rendering", "Architectural", "Cinematic", "Contemporary", "Cosmic", "Decorative",
    "Environmental", "Geometric", "Historic", "Hyperrealism", "Industrial", "Kinetic", "Landscape",
    "Monochrome", "Narrative", "Naturalistic", "Panoramic", "Photorealism", "Portraiture", "Psychedelic", "Romantic",
    "Rustic", "Scenic", "Scientific", "Urban", "Virtual",
    "Surrealist Pop", "Gothic Revival", "Mid-Century Modern", "Ukiyo-e", "Postmodernism", "Social Realism", "Constructivism", 
    "Precisionism", "Outsider Art", "Naïve Art", "Folk Art", "Abstract Expressionism", "Pop Surrealism", "Cyberpunk", 
    "Steampunk", "Lowbrow Art", "Visionary Art", "Conceptual Art", "Performance Art", "Installation Art", 
    "Assemblage", "Metaphysical Art", "Existential Art", "Bio Art", "Hypermodernism", "Transhuman Art", 
    "Retro Futurism", "Magic Realism", "Pseudorealism", "Hypernaturalism", "Ecological Art", "Sound Art",
    "Soundwave Art", "Glitch Art", "Algorithmic Art", "Interactive Art", "Virtual Reality Art", "Augmented Reality Art",
    "Net Art", "Generative Art", "AI Art", "Art Brut", "Earth Art", "Land Art", "Sustainable Art", "Green Art"
]


outfit_list = [
    "Police outfit", "Wizard outfit", "Pirate outfit", "Superhero outfit", "Doctor outfit",
    "Cowboy outfit", "Knight outfit", "Princess outfit", "Chef outfit", "Astronaut outfit",
    "Detective outfit", "Samurai outfit", "Vampire outfit", "Clown outfit", "Firefighter outfit",
    "Ninja outfit", "Robot outfit", "Pirate captain outfit", "Army outfit", "Ballerina outfit",
    "Santa Claus outfit", "Rockstar outfit", "Fairy outfit", "Jester outfit", "Zookeeper outfit",
    "Steampunk outfit", "Pharaoh outfit", "Witch outfit", "Alien outfit", "Hawaiian outfit",
    "Mad scientist outfit", "Renaissance outfit", "Gladiator outfit", "Circus performer outfit", "Medieval peasant outfit",
    "Safari outfit", "Greek god/goddess outfit", "Mermaid outfit", "Gothic outfit", "Mafia outfit",
    "Samurai armor", "Victorian era outfit", "Western outlaw outfit", "Tuxedo", "Prom dress outfit",
    "School uniform", "Pirate wench outfit", "Martial artist outfit", "Hippie outfit", "80s disco outfit",
    "Racing driver outfit", "Viking outfit", "Phantom of the Opera outfit", "Sheriff outfit", "Cleric outfit",
    "Ancient Roman toga", "Explorer outfit", "Pharaoh's guard outfit", "Zombie outfit", "Ghost outfit",
    "Mummy outfit", "Pilot outfit", "Space suit", "Futuristic soldier outfit", "Bee keeper outfit",
    "Monk outfit", "Priest outfit", "Angel outfit", "Devil outfit", "Musketeer outfit",
    "Matador outfit", "Samurai warrior outfit", "Highlander outfit", "Space ranger outfit", "Alien invader outfit",
    "Chauffeur outfit", "Butler outfit", "Geisha outfit", "Indian chief outfit", "Roman centurion outfit",
    "French maid outfit", "Grecian warrior outfit", "Pirate queen outfit", "Circus ringmaster outfit", "Rodeo clown outfit",
    "Alchemist outfit", "Artisan outfit", "Druid outfit", "Barbarian outfit", "Sailor outfit",
    "Pilot's uniform", "Spartan warrior outfit", "Incan emperor outfit", "Shogun outfit", "Knight templar outfit",
    "Cyberpunk outfit", "Time traveler outfit", "Elven archer outfit", "Dark knight outfit", "Supervillain outfit",
    "Fencing outfit", "Samurai lord outfit", "Roman gladiator outfit", "Pirate ship captain outfit", "Plague doctor outfit",
    "Cowgirl outfit", "Construction worker outfit", "Lifeguard outfit", "Warrior princess outfit", "Jungle explorer outfit",
    "Dinosaur outfit", "Ski outfit", "Ice skater outfit", "Jedi outfit", "Sith Lord outfit",
    "Spaceship commander outfit", "Ancient Egyptian outfit", "Magician outfit", "Circus strongman outfit", "Renaissance fair outfit",
    "Medieval knight outfit", "Royal guard outfit", "Beekeeper suit", "Renaissance jester outfit", "Elven warrior outfit",
    "Dragon rider outfit", "Tango dancer outfit", "Diva outfit", "Space explorer outfit", "Ancient monk outfit",
    "Scottish highlander outfit", "Opera singer outfit", "Samurai general outfit", "Norse god outfit", "Wizard's apprentice outfit",
    "Sorceress outfit", "Mafia boss outfit", "Cabin crew outfit", "Gymnast outfit", "Underwater diver outfit",
    "Fisherman outfit", "Mountaineer outfit", "Biker outfit", "Skateboarder outfit", "Sports referee outfit",
    "Baseball player outfit", "Football player outfit", "Basketball player outfit", "Boxer outfit", "Karate uniform",
    "K-pop star outfit", "Rock band member outfit", "Pop singer outfit", "Reggae outfit", "Country singer outfit",
    "Anime character outfit", "Supermodel outfit", "Opera diva outfit", "Martian explorer outfit", "Cyborg outfit",
    "Dragon costume", "Knight's squire outfit", "Celtic warrior outfit", "Gladiator champion outfit", "Samurai archer outfit",
    "Roman senator outfit", "Tudor monarch outfit", "French revolutionary outfit", "Egyptian queen outfit", "Aztec warrior outfit",
    "Ancient Chinese emperor outfit", "Victorian butler outfit", "French renaissance outfit", "Italian gondolier outfit", "Spanish flamenco dancer outfit",
    "Hawaiian hula dancer outfit", "Bollywood dancer outfit", "Traditional Japanese kimono", "African tribal warrior outfit", "Australian bushman outfit"
]
outfit_list=list(set(outfit_list))

shape_list=[
    "Cube", "Round", "Oval", "Sphere", "Triangle", "Rectangle", "Cylinder", "Ellipse", "Pyramid", "Hexagon",
    "Pentagon", "Diamond", "Star", "Cone", "Disc", "Heart", "Teardrop", "Bell", "Spiral", "Hourglass",
    "Arrow", "Crescent", "Bullet", "Torus", "Helix", "Wedge", "Prism", "Pear", "Dome", "Fan",
    "Oblong", "Parallelogram", "Rhombus", "Trapezoid", "Quadrilateral", "Octagon", "Nonagon", "Decagon", "Ring", "Blade",
    "Lattice", "Cross", "Petal", "Wave", "Loop", "Arch", "Crescent", "Shell", "Clover", "Chevron",
    "Zigzag", "Knot", "Grid", "Vortex", "Pill", "Banner", "Ribbon", "Scallop", "Loom", "Emboss",
    "Chamfer", "Channel", "Gusset", "Flute", "Ridge", "Drape", "Ledge", "Niche", "Notch", "Swoop",
    "Twist", "Fold", "Branch", "Arc", "Bow", "Reel", "Flare", "Cusp", "Cove", "Dagger",
    "Shingle", "Curl", "Funnel", "Groove", "Pocket", "Scroll", "Stitch", "Swirl", "Trap", "Zip",
    "Spindle", "Chamfer", "Plume", "Crescent", "Cross", "Facet", "Shard", "Wedge", "Bar", "Loop",
    "Hook", "Spoke", "Horn", "Band", "Twig", "Loom", "Weave", "Stalk", "Spike", "Twig",
    "Sprout", "Thread", "Whorl", "Blip", "Mound", "Blob", "Dot", "Speck", "Clump", "Mesh",
    "Fiber", "Strand", "Filament", "Ribbon", "Tangle", "Mat", "Mesh", "Fleck", "Dapple", "Fret"
]
shape_list=list(set(shape_list))
texture_list=[
            "Glossy", "Shiny", "Matte", "Smooth", "Velvety", "Silky", "Fluffy", "Soft", "Fuzzy", "Sleek",
            "Coarse", "Bristly", "Curly", "Feathery", "Wiry", "Bushy", "Slick", "Plush", "Downy", "Pearly",
            "Satiny", "Shaggy", "Woolly", "Frizzy", "Glistening", "Glossed", "Tangled", "Matted", "Thick", "Sparse",
            "Scruffy", "Frayed", "Patchy", "Sheeny", "Puffy", "Wiry", "Fine", "Shagged", "Silken", "Rippled",
            "Fluffed", "Sheared", "Tufted", "Trimmed", "Frilled", "Crisp", "Mink-like", "Velour", "Peached", "Sheeny"
            ]
texture_list=list(set(texture_list))
class CaptionGeneratorPet(CaptionGenerator):
    def __init__(self,bind_attributes=False):
        super().__init__()
        self.bind_attributes=bind_attributes
        if self.bind_attributes:
            self.types=['human_interactions','object_relations','backgrounds','styles','attributes']
        else:
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
        mlm_caption=mlm_caption.replace("<new>","{}")
        return mlm_caption
    def generate_triplet(self): # HERE
        sampled_type_pos,sampled_type_neg=np.random.choice(self.types,size=2)
        if sampled_type_pos=='human_interactions':
            anchor=self.generate_human_interactions_caption()
        elif sampled_type_pos=='object_relations':
            anchor=self.generate_object_relations_caption()
        elif sampled_type_pos=='backgrounds':
            anchor=self.generate_backgrounds_caption()
        elif sampled_type_pos=='styles':
            anchor=self.generate_styles_caption() 
        elif sampled_type_pos=='attributes':
            anchor=self.generate_attributes_caption() 

        if sampled_type_neg=='human_interactions':
            neg=self.generate_human_interactions_caption()
        elif sampled_type_neg=='object_relations':
            neg=self.generate_object_relations_caption()
        elif sampled_type_neg=='backgrounds':
            neg=self.generate_backgrounds_caption()
        elif sampled_type_neg=='styles':
            neg=self.generate_styles_caption() 
        elif sampled_type_neg=='attributes':
            neg=self.generate_attributes_caption() 
        return anchor,neg

    def generate_human_interactions_caption(self):
        interaction=np.random.choice(interactions)
        subject=np.random.choice(subjects)
        prompt=f"<new> is {interaction} a {subject}"
        return prompt
    
    def generate_object_relations_caption(self):
        rel=np.random.choice(relative_words)        
        color=np.random.choice(colors)        
        other_obj=np.random.choice(other_objs)        
        prompt=f"<new> {rel} a {color} {other_obj}"
        return prompt
    def generate_backgrounds_caption(self):
        background=np.random.choice(backgrounds)
        if np.random.rand()<0.5:
            prompt=f"<new> with the {background} in the background"
        else:
            prompt=f"A view of the <new> at {background}"
        return prompt
    def generate_styles_caption(self):
        fmt=np.random.choice(['captured','depicted','rendered'])
        style=np.random.choice(styles)
        prompt=f"<new> {fmt} in the {style} style"
        return prompt
    def generate_attributes_caption(self):

        att_type=np.random.choice(['outfit','shape','texture','color'])
        if att_type=='outfit':
            att_outfit=np.random.choice(outfit_list)
            fmts=['fmt1','fmt2','fmt3']
            fmt=np.random.choice(fmts)
            if fmt=='fmt1':
                prompt=f"a <new> in a {att_outfit}"
            elif fmt=='fmt2':
                prompt=f"a <new> wearing a {att_outfit}"
            else:
                prompt=f"a <new> dressed in a {att_outfit}"
        elif att_type=='shape':
            att_shape=np.random.choice(shape_list)
            shape_prob=np.random.rand()
            fmts=['fmt1','fmt2','fmt3','fmt4']
            fmt=np.random.choice(fmts)
            if fmt=='fmt1':
                prompt=f"a {att_shape} shaped <new>"
            elif fmt=='fmt2':
                prompt=f"a <new> with {att_shape} shape"
            elif fmt=='fmt3':
                prompt=f"a <new> in a {att_shape} shape"
            else:
                prompt=f"a <new> in a {att_shape} form"
            
        elif att_type=='texture':
            att_texture=np.random.choice(texture_list)
            texture_prob=np.random.rand()
            fmts=['fmt1','fmt2','fmt3']
            fmt=np.random.choice(fmts)
            if fmt=='fmt1':
                prompt=f"a {att_texture} <new>"
            elif fmt=='fmt2':
                prompt=f"a <new> in {att_texture} texture"
            else:
                prompt=f"a {att_texture} textured <new>"




        elif att_type=='color':
            att_color=np.random.choice(colors)
            color_prob=np.random.rand()
            if color_prob<(1/3):
                prompt=f"a {att_color} <new>"
            elif (color_prob>=1/3) and (color_prob<2/3):
                prompt=f"a <new> in {att_color} color"
            else:
                prompt=f"a {att_color} colored <new>"
        else:
            assert False,'undefined type'
        return prompt