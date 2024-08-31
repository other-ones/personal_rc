
HUMAN_INTERACTIONS = [
    "playing with", "talking to", "feeding", "hugging", "walking", "training", "petting", 
    "brushing", "cuddling with", "carrying", "photographing", "watching", "dressing up", 
    "giving a bath to", "cleaning up after", "taking to the vet", "grooming", "teaching tricks to", 
    "running with", "sleeping beside", "reading a book with", "celebrating a birthday with", 
    "sharing food with", "taking a selfie with", "protecting", "napping with", "taking for a car ride with", 
    "playing fetch with", "riding in a bicycle basket with", "playing in the park with", "resting beside", 
    "sitting on a bench with", "enjoying a sunny day with", "sharing a secret with", 
    "making a video with", "having a picnic with", "watching TV with", "doing yoga with", 
    "playing a game with", "taking a walk in the rain with", "going for a swim with", 
    "enjoying a sunset with", "having breakfast with", "dancing with", "skateboarding with", 
    "hiking with", "watching a movie with", "going on an adventure with", "swimming with"
]

HUMANS = [
    "teacher", "a soldier", "magician", "acrobat", "explorer", "scientist", "artist", "writer", "detective", "athlete",
    "nurse", "doctor", "politician", "monk", "priest", "sailor", "pilot", "chef", "farmer", "hunter", "fisherman",
    "trader", "smith", "jeweler", "tailor", "gardener", "librarian", "architect", "engineer", "programmer", "astronaut",
    "alien", "vampire", "wizard", "knight", "samurai", "ninja", "king", "queen", "prince", "princess", "bard", "mercenary",
    "cyborg", "robot", "android", "superhero", "villain", "Albert Einstein", "Marie Curie", "Leonardo da Vinci",
    "Cleopatra", "William Shakespeare", "Mahatma Gandhi", "Winston Churchill", "Amelia Earhart", "Abraham Lincoln",
    "Martin Luther King Jr.", "Frida Kahlo", "Pablo Picasso",  "Stephen Hawking",
    "Beethoven", "Mozart", "Vincent van Gogh","Isaac Newton", "Galileo Galilei",
    "Marco Polo", "Socrates", "Aristotle", "Tutankhamun", "Queen Elizabeth", "Genghis Khan",
]
ANIMALS = [
    "dog", "cat", "bear", "cow", "pig", "horse", "sheep", "goat", "rabbit", "deer",
    "elephant", "lion", "tiger", "wolf", "fox", "panda", "zebra", "giraffe", "kangaroo",
    "monkey", "donkey", "buffalo", "camel", "hedgehog", "koala", "otter", "squirrel",
    "llama", "cheetah", "leopard", "raccoon", "skunk"
]

COLORS = [
    "red", "blue", "green", "yellow", "black", "white",
    "orange", "purple", "gray", "brown", "silver", "gold",
    "beige", "ivory", "teal", "navy", "maroon", "turquoise",
    "lime", "charcoal", "coral", "cyan", "magenta", "olive",
]

NONLIVINGS = [
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

RELATIVES = [
    "beside", "near", "next to", "behind", "in front of", "adjacent to", 
    "above", "below", "alongside", "atop", "against", "surrounding",
    "beneath", "over", "under", "to the left of", "to the right of",
    "on top of",  
    # "suspended above",
    # "encircled by", "interspersed with", "between", "flanked by","aligned with", "sprawled in front of", 
    #  "arrayed around"
    ]

BACKGROUNDS = [
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

STYLES = [
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







WEARINGS=[
"superhero costume","tuxedo","holiday sweater","raincoat","Halloween costume",
"Christmas sweater","pirate costume","dinosaur costume","bee costume","lion costume",
"wizard robe","knitted sweater","biker jacket","sports jersey","Santa Claus suit",
"ballet tutu","military uniform","princess dress","leather jacket","elf costume",
"witch costume","snow boots","denim jacket","pajamas","bow tie",
"scarf and hat","tropical shirt","hula skirt","soccer uniform","baseball cap",
"onesie","flannel shirt","rain boots","aviator jacket","fairy wings",
"clown costume","artist beret","graduate cap and gown","magician's cape","garden overalls",
"puffer jacket","floral dress","tank top","ski suit","vampire costume",
"pumpkin costume","football helmet","astronaut suit","bunny ears","sunglasses and bandana",
"swimsuit","tartan kilt","dinosaur onesie","rain poncho","king's robe",
"medieval armor","tiger costume","hooded sweatshirt","vest","bandana",
"shirt","t-shirt","puffer vest","shorts","apron",
"cape","parka","baseball jersey","harness","tracksuit",
"overalls","gloves","turtleneck","zip-up jacket","peacoat",
"snowsuit","blazer","sneakers","overcoat","long coat",
"fleece jacket","winter hat","earmuffs","towel wrap","long sleeve shirt",
"hoodie with ears","flannel pajamas","college sweater","gym shirt","fancy dress",
"velvet jacket","swim trunks","plaid jacket","Christmas hat","bandana with a print",
"overalls","wizard hat","striped shirt","raincoat with hood","denim overalls",
"scarf","fluffy scarf","tie-dye shirt","rainbow sweater","v-neck shirt",
"button-up shirt","poncho","suspenders","striped pajamas","sun hat",
"fur-lined vest","ski jacket","snuggly robe","hiking boots","mesh vest",
"safari hat","striped onesie","pullover","knit hat","padded jacket",
"cape with a hood","wrap-around coat","parka","beanie","plaid shirt",
"beach shirt","fleece-lined jacket","festival shirt","hiking vest","summer shirt",
"spring jacket","rainproof jacket","sunglasses","puffy jacket","fall sweater",
"plaid shirt","turtleneck sweater","leather vest","winter jacket","light summer hoodie",
"zip-up fleece","sweater with stripes","gym vest","sporty jacket","windbreaker",
"knit sweater","sweatpants","camouflage hoodie","quilted jacket","vintage coat",
"tactical vest","light raincoat","high-visibility jacket","knitted jumper","boots",
"hooded poncho","overcoat","short sleeve shirt","blazer","slippers",
"waterproof jacket","sombrero","costume","superhero costume","tuxedo",
"holiday sweater","raincoat","Halloween costume","Christmas sweater","pirate costume",
"dinosaur costume","bee costume","lion costume","wizard robe","knitted sweater",
"biker jacket","sports jersey","Santa Claus suit","ballet tutu","military uniform",
"princess dress","leather jacket","elf costume","witch costume","snow boots",
"denim jacket","pajamas","tropical shirt","hula skirt","soccer uniform",
"hoodie","onesie","flannel shirt","rain boots","aviator jacket",
"clown costume","graduate cap and gown","magician's cape","garden overalls","puffer jacket",
"floral dress","tank top","ski suit","vampire costume","pumpkin costume",
"astronaut suit","tartan kilt","dinosaur onesie","rain poncho","king's robe",
"medieval armor","tiger costume","hooded sweatshirt","vest","puffer vest",
"cape","parka","baseball jersey","tracksuit","overalls",
"zip-up jacket","peacoat","snowsuit","overcoat","long coat",
"fleece jacket","towel wrap","long sleeve shirt","hoodie with ears","flannel pajamas",
"college sweater","gym shirt","fancy dress","velvet jacket","swim trunks",
"plaid jacket","bandana with a print","overalls","striped shirt","raincoat with hood",
"denim overalls","striped pajamas","fur-lined vest","ski jacket","snuggly robe",
"padded jacket","cape with a hood","wrap-around coat","rain boots","parka",
"fleece-lined jacket","festival shirt","hiking vest","summer shirt","spring jacket",
"rainproof jacket","puffy jacket","fall sweater","plaid shirt","turtleneck sweater",
"leather vest","winter jacket","light summer hoodie","zip-up fleece","sweater with stripes",
"gym vest","sporty jacket","windbreaker","knit sweater","sweatpants",
"camouflage hoodie","quilted jacket","vintage coat","tactical vest","light raincoat",
"high-visibility jacket","knitted jumper","hooded poncho","overcoat","waterproof jacket",
]
OUTFITS=[
"police outfit","firefighter outfit","sailor outfit","cowboy outfit","doctor outfit",
"chef outfit","prince outfit","lifeguard outfit","camouflage outfit","sheriff outfit",
"nurse outfit","sporting outfit","police outfit","firefighter outfit","sailor outfit",
"cowboy outfit","doctor outfit","chef outfit","prince outfit","lifeguard outfit",
"camouflage outfit","sheriff outfit","nurse outfit","sporting outfit","construction outfit",
]

NONVLIVING_INTERACTIONS_PASSIVE = [
    'inspected by', 'examined by','repaired by', 'sketched by', 
    'cleaned by', 'polished by', 'photographed by',
    'packaged by', 'analyzed by', 'painted by', 
    'recorded by', 'filmed by', 'transported by',
    'restored by', 'evaluated by',
    'auctioned by',  'implemented by','researched by','engraved by', 'modified by', 
    'presented by', 'juggled by', 'assembled by','scanned by', 'protected by',
    'carved into ice by','carved into wood by','carved into marble by','carved into stone by',
]

NONVLIVING_INTERACTIONS_ACTIVE = [
    'inspecting', 'examining', 'repairing', 'sketching', 
    'cleaning', 'polishing', 'photographing', 
    'packaging', 'analyzing', 'painting', 
    'recording', 'filming', 'transporting', 
    'restoring', 'evaluating', 
    'auctioning','implementing', 'researching', 'engraving', 'modifying', 
    'giving a presentation on', 'juggling',  'assembling', 
    'scanning', 'protecting',
    'carving ice into','carving wood into', 'carving marble into','carving stone into'
]
