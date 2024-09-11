import re
import random
import time
import cv2
count=0
from shapely.geometry import Polygon
import numpy as np
import torch
from torchvision import transforms
from pathlib import Path
import pdb
from torch.utils.data import Dataset
import os
from PIL import Image,ImageDraw
import string
Image.MAX_IMAGE_PIXELS = 1000000000
alphabet = string.digits + string.ascii_lowercase + string.ascii_uppercase + string.punctuation + ' ' # len(aphabet) = 95
alphabet_dic = {}
for index, c in enumerate(alphabet):
    alphabet_dic[c] = index + 1 # the index 0 stands for non-character

mask_transforms = transforms.Compose(
    [
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),

    ]
)
roi_mask_transforms = transforms.Compose(
    [
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),

    ]
)

templates=prompts = [
    "A whimsical {} floating in a bubble",
    "A magical {} granting wishes in a fairy tale kingdom",
    "A steampunk {} powering a fantastical machine",
    "An ancient {} adorned with mystical runes",
    "A legendary {} featured in folklore",
    "A cyberpunk {} navigating a dystopian cityscape",
    "A time-traveling {} lost in the past",
    "A futuristic {} exploring uncharted galaxies",
    "A haunted {} lurking in a spooky graveyard",
    "A celestial {} shining brightly in the night sky",
    "A medieval {} guarding a castle entrance",
    "A cosmic {} drifting through the cosmos",
    "A robotic {} assisting humans in the future",
    "A fantasy {} living in a magical realm",
    "A mystical {} hidden deep within a forest",
    "A legendary {} sought after by treasure hunters",
    "A whimsical {} found in a dream",
    "A futuristic {} featured in a sci-fi movie",
    "An enchanted {} protected by a powerful spell",
    "A mysterious {} discovered in an ancient tomb",
    "A celestial {} worshipped by ancient civilizations",
    "A mythical {} roaming the earth in ancient times",
    "An otherworldly {} encountered in a parallel universe",
    "A legendary {} hidden in the depths of the ocean",
    "A magical {} appearing in a children's storybook",
    "A cosmic {} observed through a telescope",
    "A futuristic {} depicted in a science fiction novel",
    "A mystical {} revealed in a prophetic vision",
    "A legendary {} immortalized in ancient mythology",
    "A celestial {} observed by astronomers",
    "A mythical {} depicted in ancient cave paintings",
    "An otherworldly {} encountered during astral projection",
    "A magical {} protected by ancient guardians",
    "A futuristic {} designed by advanced AI",
    "A mythical {} rumored to possess mystical powers",
    "A celestial {} admired by stargazers",
    "A legendary {} featured in epic tales",
    "An ancient {} discovered by archaeologists",
    "A mystical {} summoned by a sorcerer",
    "A mythical {} worshipped as a deity",
    "A celestial {} shining brightly in the night sky",
    "A legendary {} sought after by treasure hunters",
    "A mythical {} roaming the earth in ancient times",
    "A whimsical {} found in a dream",
    "A celestial {} observed by astronomers",
    "A legendary {} immortalized in ancient mythology",
    "A mystical {} revealed in a prophetic vision",
    "An otherworldly {} encountered during astral projection",
    "A magical {} protected by ancient guardians",
    "A futuristic {} designed by advanced AI",
    "A mythical {} rumored to possess mystical powers",
    "A celestial {} admired by stargazers",
    "A legendary {} featured in epic tales",
    "An ancient {} discovered by archaeologists",
    "A mystical {} summoned by a sorcerer",
    "A mythical {} worshipped as a deity",
]
nouns =["Airplane", "Alien", "Alligator", "Ant", "Antelope", "Astronaut", "Bass", "Bat", "Bee", "Beetle", "Bicycle", "Bird", "Boat", "Book", "Bridge", "Building", "Bus", "Butterfly", "Car", "Cat", "Caterpillar", "Chair", "Chicken", "Clam", "Cloud", "Cod", "Cow", "Crab", "Crocodile", "Crow", "Deer", "Desk", "Dog", "Dolphin", "Dragon", "Duck", "Dwarf", "Eagle", "Eel", "Elephant", "Elf", "Elk", "Fairy", "Falcon", "Fish", "Flower", "Fly", "Fox", "Frog", "Galaxy", "Ghost", "Giant", "Giraffe", "Goat", "Goblin", "Goose", "Gorilla", "Hamster", "Hawk", "Hedgehog", "Helicopter", "Hen", "Hero", "Horse", "House", "Insect", "Jellyfish", "Kangaroo", "Koala", "Ladybug", "Lake", "Lion", "Lizard", "Lobster", "Mackerel", "Mermaid", "Monster", "Moon", "Moose", "Mosquito", "Motorcycle", "Mountain", "Mouse", "Ocean", "Octopus", "Orc", "Owl", "Oyster", "Panda", "Paper", "Parrot", "Pen", "Pencil", "Penguin", "Phone", "Pig", "Pigeon", "Planet", "Puffin", "Rabbit", "Raccoon", "Rat", "Raven", "River", "Robot", "Rocket", "Rooster", "Salmon", "Sea", "Seagull", "Seal", "Shark", "Sheep", "Ship", "Shrimp", "Skunk", "Sky", "Slug", "Snail", "Snake", "Spider", "Squid", "Squirrel", "Star", "Starfish", "Submarine", "Sun", "Superhero", "Swan", "Table", "Television", "Tiger", "Train", "Tree", "Troll", "Trout", "Truck", "Tuna", "Turtle", "Vampire", "Villain", "Wasp", "Werewolf", "Whale", "Witch", "Wizard", "Wolf", "Zebra", "Zombie"]
class NounDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """
    def __init__(
            self,
            caption_path='generated/captions.txt',
            nsamples='generated/captions.txt'
    ):
        caption_lines=open(caption_path).readlines()
        self.fnames=[]
        self.captions=[]
        self.nouns=[]
        for line in caption_lines:
            fname,caption,noun=line.strip().split('\t')
            self.fnames.append(fname)
            self.captions.append(caption)
            self.nouns.append(noun)
        
        self.num_instance_images=len(caption_lines)

    def __len__(self):
        return self.num_instance_images#len(self.db_list)
    def __getitem__(self, index):
        example = {}
        noun=self.nouns[index]
        fname=self.fnames[index]
        caption=self.captions[index]
        example['caption']=caption
        example['noun']=noun
        example['fname']=fname
        return example
 