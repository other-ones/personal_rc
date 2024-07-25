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
    "Elon Musk","Oprah Winfrey", "Leonardo DiCaprio","Beyoncé","Cristiano Ronaldo","Taylor Swift",
    "Bill Gates","Rihanna","Angelina Jolie","LeBron James",
    "Ariana Grande","Lionel Messi","Kim Kardashian","Barack Obama",
    "Justin Bieber","Tom Hanks","Selena Gomez","Jennifer Lopez",
    "Emma Watson","Brad Pitt","Lady Gaga","Robert Downey Jr.",
    "Kanye West","Will Smith","Albert Einstein","Stephen Hawking",
    "Isaac Newton","Marie Curie","Mahatma Gandhi","Nelson Mandela",
    "Martin Luther King Jr.","Walt Disney","Steve Jobs",
    "Mark Zuckerberg","Charles Darwin","Pablo Picasso",
    "Vincent van Gogh","William Shakespeare","Ludwig van Beethoven",
    "Wolfgang Amadeus Mozart","Michael Jackson","Freddie Mercury",
    "Madonna","Elvis Presley","Marilyn Monroe",
    "J.K. Rowling","Stephen King","Ernest Hemingway",
    "Jane Austen","George Washington","Napoleon Bonaparte",
    "Winston Churchill","Mother Teresa","John F. Kennedy",
    "Isaac Asimov","Nikola Tesla","Thomas Edison",
    "Galileo Galilei"]
subjects=list(set(subjects))
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
interactions=list(set(interactions))
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
other_objs=list(set(other_objs))
colors = [
    "red", "blue", "green", "yellow", "black", "white",
    "orange", "purple", "gray", "brown", "silver", "gold",
    "beige", "ivory", "teal", "navy", "maroon", "turquoise",
    "lime", "charcoal", "coral", "cyan", "magenta", "olive",
]
colors=list(set(colors))

relative_words = [
    "beside", "near", "next to", "behind", "in front of",
    "above", "below", "across from", "alongside", "amid",
    "atop", "against", "within", "surrounding",
    "beneath", "over", "under", "to the left of", "to the right of",
    "on top of", "hanging over", "leaning on", "framed by",
    "encircled by", "interspersed with", "between", "flanked by",
    "opposite", "aligned with", "adjacent to", 
    "sprawled in front of", "nestled among", "mounted on",
    "clinging to", "perched on", "suspended above", "arrayed around",
    "bordering"
]
relative_words=list(set(relative_words))
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
relative_words=list(set(relative_words))

attributes = [
    # Colors
    "red", "blue", "green", "yellow", "brown", "white", "black", "gray", "orange", "purple", "pink", "golden", "silver",
    "beige", "ivory", "teal", "navy", "maroon", "turquoise",
    "lime", "charcoal", "coral", "cyan", "magenta", "olive",
    # Textures
    "smooth", "rough", "shiny", "matte", "glossy", "textured", "polished", "bumpy", "gritty", "furry", "slick", "coarse",
    # Materials
    "wooden", "metallic", "plastic", "glass", "ceramic", "stone", "concrete", "cloth", "paper", "rubber", "leather",
    # Shapes
    "circular", "square", "rectangular", "triangular", "oval", "round", "cylindrical", "hexagonal", "octagonal", "spherical",
    # Additional Attributes
    "small", "large", "tiny", "huge", "old", "new", "antique", "modern", "ornate", "simple", "heavy", "lightweight", 
    "transparent", "opaque", "fragile", "sturdy", "decorative", "functional", "handmade", "mass-produced"
]
attributes=list(set(attributes))

class CaptionGeneratorSunglasses(CaptionGenerator):
    def __init__(self):
        super().__init__()
        self.types=['human_interactions','object_relations','backgrounds','styles']
    def generate_caption(self):
        sampled_type=np.random.choice(self.types)
        if sampled_type=='human_worn':
            mlm_caption=self.generate_human_worn_caption()
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
            anchor=self.generate_human_worn_caption()
        elif sampled_type_pos=='object_relations':
            anchor=self.generate_object_relations_caption()
        elif sampled_type_pos=='backgrounds':
            anchor=self.generate_backgrounds_caption()
        elif sampled_type_pos=='styles':
            anchor=self.generate_styles_caption() 

        if sampled_type_neg=='human_interactions':
            neg=self.generate_human_worn_caption()
        elif sampled_type_neg=='object_relations':
            neg=self.generate_object_relations_caption()
        elif sampled_type_neg=='backgrounds':
            neg=self.generate_backgrounds_caption()
        elif sampled_type_neg=='styles':
            neg=self.generate_styles_caption() 

        return anchor,neg
    def generate_human_worn_caption(self):
        subject=np.random.choice(subjects)
        if np.random.rand()<0.5:
            prompt=f"{subject} wearing <new>"
        else:
            prompt=f"{subject} wearing <new>"
        return prompt
    
    def generate_object_relations_caption(self):
        rel=np.random.choice(relative_words)        
        attr=np.random.choice(attributes)        
        other_obj=np.random.choice(other_objs)        
        prompt=f"<new> {rel} a {attr} {other_obj}"
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