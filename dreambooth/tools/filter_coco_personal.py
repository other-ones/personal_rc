import os
import numpy as np
import json

caption_path='/data/dataset/coco/karpathy/coco_caption_raw.txt'
lines=open(caption_path)
targets=[
        "cat", "dog", "sunglasses", "vase","backpack",
        "barn","flower","chair","teddybear","a pot"
         ]

# Placeholder for the object
object_placeholder = "[OBJECT]"
# Expanded lists for themes, times of day, weather conditions, and seasons
themes = [
    "a dense forest", "a city skyline", "a sunny beach", "a mountain range", "a desert", "a lake",
    "a riverbank", "a downtown area", "a suburban neighborhood", "a rural landscape", "an industrial area",
    "a historic town", "a futuristic cityscape", "an alien landscape", "a magical realm", "an abandoned village",
    "a snowy field", "a blooming garden", "a secluded coastline", "ancient ruins", "a festive marketplace",
    "a frozen tundra", "a windy plain", "a volcanic landscape", "a jungle", "a cliff", "a bustling port",
    "a quiet pond", "a starry sky", "a muddy marshland", "a grassy meadow", "a steep canyon", "a quiet forest",
    "a coral reef", "a busy street", "a tranquil park", "a moonlit valley", "a golden wheat field", "a dark cave",
    "a crowded stadium", "a broad highway", "a small island", "a tall skyscraper", "a rapid stream", "a narrow alley",
    "a clear glade", "a quiet bayou", "a dense vineyard", "a sprawling estate", "a rocky shore", "a bamboo grove",
    "a snowy pass", "a sunken valley", "a pine forest", "a sandy shore", "a brick-lined alley", "a glittering city",
    "a foggy moor", "a rolling hill", "a lush valley", "a parched plain", "a vibrant orchard", "a quiet cemetery",
    "a bustling cafe", "a serene monastery", "a large dam", "a derelict factory", "a bustling marketplace", "a high bridge",
    "a deep ocean trench", "a remote cabin", "a busy gym", "a traditional dojo", "a sprawling ranch", "a large sand dune"
]
times_of_day = [
    "at dawn", "in the morning", "at noon", "in the afternoon", "at dusk", "during twilight",
    "at night", "at midnight", "just before sunrise", "right after sunset", "in the early night",
    "in the late morning", "at the daybreak", "when the sun peaks", "as the sun sets", "under the midday sun",
    "in the early dawn", "at the evening", "before nightfall", "as daylight fades", "at the golden hour",
    "during the blue hour", "at first light", "at last light", "under the noonday sun", "in the dead of night",
    "at early twilight", "at late dusk", "just before midnight", "in the full moon light"
]
weather_conditions = [
    "on a sunny day", "under overcast skies", "during light rain", "amidst a storm", "in fog",
    "under snowfall", "with strong winds", "in haze", "in mist", "during heat", "in frost",
    "under clear skies", "during drizzle", "in humid weather", "in freezing conditions", "with light snow",
    "with heavy rain", "in scorching weather", "with mild weather", "in cool weather", "with gusty winds",
    "during a heatwave", "in a cold snap", "under a thunderstorm", "during a sleet shower", "in pelting hail",
    "with torrential rain", "in stifling humidity", "with whipping sand", "in icy conditions", "under the aurora",
    "with falling leaves", "during a rainbow", "in electrical storms", "with thick cloud cover"
]
seasons = [
    "in spring", "in summer", "in autumn", "in winter", "during the wet season", "during the dry season",
    "in the monsoon season", "in the harvest season", "in the planting season", "in the holiday season",
    "in the blooming season", "in the falling leaves season", "during the frost season", "in the growing season",
    "during the migration season", "in the nesting season"
]
# Generate captions by combining elements from each list with different formats
prefixes=[
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

captions = []
for theme in themes:
    for time in times_of_day:
        for weather in weather_conditions:
            background = f"{theme}"
            # simple = f"A picture of {object_placeholder} with {background} in the background."
            suffix = f"{object_placeholder} with {theme} in the background."
            suffix2 = f"{object_placeholder} in {theme}."
            suffix3= f"{object_placeholder} in {theme} {weather}."
            sampled_suffix=np.random.choice([suffix,suffix2,suffix3])
            sampled_prefix=np.random.choice(prefixes)
            simple=sampled_prefix.format(sampled_suffix)

            # detailed = f"{object_placeholder} seen with {background} in the background during {time} on {weather}."
            # narrative = f"As the {time} light casts shadows, {object_placeholder} stands against {background}, {weather}."
            # poetic = f"Beneath {weather} skies of {time}, {object_placeholder} resides within {background}."
            # informal = f"Here’s {object_placeholder}, chilling in {background} {time}, {weather}."
            # print(simple,'simple')
            # print(detailed,'detailed')
            captions.append(simple)
            # captions.extend([simple, detailed, narrative, poetic, informal])
print(len(captions))
np.random.shuffle(captions)
for item in captions[:10]:
    print(item)