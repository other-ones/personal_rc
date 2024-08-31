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

interactions_human = [
    "playing with", "talking to", "feeding", "hugging", "walking with", "training", "petting", 
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
]

relative_objects = [
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

interactions_pet = [
    '{} [JUNCTION] {} playing with a ball',
    '{} [JUNCTION] {} chasing each other around the garden',
    '{} [JUNCTION] {} sleeping side by side on the couch',
    '{} [JUNCTION] {} eating from adjacent bowls in the kitchen',
    '{} [JUNCTION] {} tussling gently on the living room rug',
    '{} [JUNCTION] {} licking each other’s fur clean',
    '{} [JUNCTION] {} sharing a pillow during naptime',
    '{} [JUNCTION] {} looking out the window at passing cars',
    '{} [JUNCTION] {} sitting on opposite ends of the sofa',
    '{} [JUNCTION] {} running through the hallway chasing a laser pointer',
    '{} [JUNCTION] {} sitting under the same chair seeking shade',
    '{} [JUNCTION] {} touching noses as a greeting',
    '{} [JUNCTION] {} bumping heads while playing',
    '{} [JUNCTION] {} jumping over one another in the yard',
    '{} [JUNCTION] {} eating from the same bowl when one is finished',
    '{} [JUNCTION] {} lying on a shared bed during a cold night',
    '{} [JUNCTION] {} following one another up the stairs to bed',
    '{} [JUNCTION] {} sniffing each other as a form of introduction',
    '{} [JUNCTION] {} resting in the same crate while traveling',
    '{} [JUNCTION] {} competing for a toy during playtime',
    '{} [JUNCTION] {} waiting side by side at the door to go out',
    '{} [JUNCTION] {} barking and meowing in response to each other',
    '{} [JUNCTION] {} stealing toys from each other playfully',
    '{} [JUNCTION] {} guarding their toys from each other',
    '{} [JUNCTION] {} walking together on a leash',
    '{} [JUNCTION] {} sitting together in the backyard',
    '{} [JUNCTION] {} watching the sunset from the porch',
    '{} [JUNCTION] {} digging holes together in the garden',
    '{} [JUNCTION] {} hiding from each other during play',
    '{} [JUNCTION] {} resting their heads on each other while sleeping',
    '{} [JUNCTION] {} pawing at each other playfully',
    '{} [JUNCTION] {} running up and down the stairs',
    '{} [JUNCTION] {} chasing shadows together',
    '{} [JUNCTION] {} trying to catch the same butterfly',
    '{} [JUNCTION] {} lounging in the sun together',
    '{} [JUNCTION] {} playing hide and seek',
    '{} [JUNCTION] {} competing for attention',
    '{} [JUNCTION] {} tangled in a blanket',
    '{} [JUNCTION] {} alert to a strange noise',
    '{} [JUNCTION] {} squabbling over a spot on the couch',
    '{} [JUNCTION] {} engaged in a staring contest',
    '{} [JUNCTION] {} grooming each other',
    '{} [JUNCTION] {} sharing a treat',
    '{} [JUNCTION] {} trying to outdo each other in a race',
    '{} [JUNCTION] {} nudging each other affectionately',
    '{} [JUNCTION] {} trying to fit into the same box',
    '{} [JUNCTION] {} play-fighting',
    '{} [JUNCTION] {} exploring the attic together',
    '{} [JUNCTION] {} causing mischief in the kitchen',
    '{} [JUNCTION] {} taking turns chasing their tails',
    '{} [JUNCTION] {} playing with a water hose'
]

relative_pets = [
    '{} next to {}',
    '{} behind {}',
    '{} in front of {}',
    '{} beside {}',
    '{} curled around {}',
    '{} under {}',
    '{} leaning on {}',
    '{} facing {}',
    '{} apart from {}',
    '{} across from {}',
    '{} nose to nose with {}',
    '{} circling {}',
    '{} under the chin of {}',
    '{} back-to-back with {}',
    '{} following {}',
    '{} guarding {}',
    '{} near {}',
    '{} around {}',
    '{} head-to-toe with {}',
    '{} intertwined with {}',
    '{} on top of {}',
    '{} tucked under {}',
    '{} sandwiched between {} and the sofa',
    '{} peeking over {}',
    '{} nestled against {}',
    '{} sprawled near {}',
    '{} resting its head on {}',
    '{} playfully pouncing towards {}',
    '{} lying down opposite {}',
    '{} watching {} quietly',
    '{} jumping beside {}',
    '{} play-bowing to {}',
    '{} sniffing {} curiously',
    '{} sprinting ahead of {}',
    '{} sneaking up on {}',
    '{} hiding behind {}',
    '{} waiting for {} by the door',
    '{} keeping a close eye on {}',
    '{} trailing {} by a whisker',
    '{} crouched near {}',
    '{} lounging across from {}',
    '{} perched above {}',
    '{} settling down beside {}'
]


class CaptionGeneratorTwoPets(CaptionGenerator):
    def __init__(self):
        super().__init__()
        self.types=['human_interactions','object_relations','backgrounds','styles','pet_interactions','pet_relations']
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
        elif sampled_type=='pet_interactions':
            mlm_caption=self.generate_pet_interactions_caption()
        elif sampled_type=='pet_relations':
            mlm_caption=self.generate_pet_relations_caption()
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
        elif sampled_type_pos=='pet_interactions':
            anchor=self.generate_pet_interactions_caption()
        elif sampled_type_pos=='pet_relations':
            anchor=self.generate_pet_relations_caption()

        if sampled_type_neg=='human_interactions':
            neg=self.generate_human_interactions_caption()
        elif sampled_type_neg=='object_relations':
            neg=self.generate_object_relations_caption()
        elif sampled_type_neg=='backgrounds':
            neg=self.generate_backgrounds_caption()
        elif sampled_type_neg=='styles':
            neg=self.generate_styles_caption() 
        elif sampled_type_neg=='pet_interactions':
            neg=self.generate_pet_interactions_caption()
        elif sampled_type_neg=='pet_relations':
            neg=self.generate_pet_relations_caption()
        return anchor,neg
    

    def generate_pet_relations_caption(self):
        prompt=np.random.choice(relative_pets)
        return prompt
    def generate_pet_interactions_caption(self):
        prompt=np.random.choice(interactions_pet)
        return prompt
    def generate_human_interactions_caption(self):
        interaction=np.random.choice(interactions_human)
        subject=np.random.choice(subjects)
        prompt=f"<new> are {interaction} a {subject}"
        prompt=prompt.replace("<new>","{} [JUNCTION] {}")
        return prompt
    
    def generate_object_relations_caption(self):
        rel=np.random.choice(relative_objects)        
        color=np.random.choice(colors)        
        other_obj=np.random.choice(other_objs)        
        prompt=f"<new> {rel} a {color} {other_obj}"
        prompt=prompt.replace("<new>","{} [JUNCTION] {}")
        return prompt
    def generate_backgrounds_caption(self):
        background=np.random.choice(backgrounds)
        if np.random.rand()<0.5:
            prompt=f"<new> with the {background} in the background"
        else:
            prompt=f"A view of the <new> at {background}"
        prompt=prompt.replace("<new>","{} [JUNCTION] {}")
        return prompt
    def generate_styles_caption(self):
        fmt=np.random.choice(['captured','depicted','rendered'])
        style=np.random.choice(styles)
        prompt=f"<new> {fmt} in the {style} style"
        prompt=prompt.replace("<new>","{} [JUNCTION] {}")
        return prompt