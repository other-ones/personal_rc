import torchvision
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
import PIL
from PIL import Image,ImageDraw
import string
import albumentations as A
from packaging import version

# Added
# Added
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
BACKGROUNDS=[
    "city", "mountain", "blue house", "tree and autumn leaves", "river", 
    "beach", "forest", "desert", "waterfall", "field of flowers", 
    "snowy landscape", "sunset", "sunrise", "garden", "bridge", 
    "lake", "meadow", "rainbow", "starry night", "castle", 
    "windmill", "lighthouse", "bustling street", "skyline", 
    "foggy morning", "canyon", "village", "farm", "valley", 
    "jungle", "waterfall", "rocky shore", "harbor", "dock",
    "vineyard", "mountain range", "glacier", "cliff", "temple",
    "clock tower", "skyscraper", "tower", "pier", "forest path",
    "mountain peak", "train track", "tunnel", "riverbank", 
    "field of wheat", "snow-covered tree", "moonlit night", "distant island",
    "cobblestone street", "desert dune", "grassy hill", "quaint cottage",
    "fishing boat", "pagoda", "volcano", "rainforest", "sunflower field",

    "Great Wall of China", "Times Square", "botanic garden",
    "Sahara Desert", "coral reef", "Mount Everest base",
    "Amazon rainforest", "Venetian canal", "Paris cafe",
    "modern art museum", "Silicon Valley tech lab", "Egyptian pyramid",
    "Hollywood movie set", "medieval castle", "Tokyo skyline",
    "Antarctic ice field", "Caribbean beach", "international space station",
    "fairy tale castle", "major city public library", "Renaissance chapel",
    "Roman Colosseum", "Grand Canyon", "Eiffel Tower",
    "Mount Fuji", "Niagara Falls", "Yellowstone National Park", "Lake Tahoe", "Great Barrier Reef",
    "Kyoto temple", "Taj Mahal", "Golden Gate Bridge", "Machu Picchu", "Victoria Falls",
    "Santorini cliffside", "Milan fashion district", "Berlin Wall", "Sydney Opera House",
]
if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }

class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        include_prior_concept,
        size=512,
        interpolation="bicubic",
        flip_p=0,
        center_crop=False,
        exclude_suffix=True,
        mlm_target='all',
        mask_prob=0.25,
        mask_token_ids=None,
        get_images=True,
        prompt_type=None,
        train_prior_concept=None,
        placeholder_token=None,
        caption_path=None,
        target=None,
        prior_only=None,
        num_mask=0
    ):
        self.num_mask=num_mask
        self.prior_only=prior_only
        self.prompt_type=prompt_type
        self.include_prior_concept=include_prior_concept
        # self.captions=open(caption_path).readlines()
        if include_prior_concept:
            self.placeholder='{} {}'.format(placeholder_token,train_prior_concept)
        else:
            self.placeholder='{}'.format(placeholder_token)
        self.captions_raw=[]
        self.captions=[]
        self.captions_simple=[]
        self.captions_simple_raw=[]
        for bg in BACKGROUNDS:
            # bg_words=[]
            # bg_splits=bg.split()
            # for item in bg.split():
            #     bg_word='<bg>{}'.format(item)
            #     bg_words.append(bg_word)
            # bg_new=' '.join(bg_words)
            bg_new='<bg>{}'.format(bg)
            if prior_only:
                caption="a picture of {} with {} in the background".format(train_prior_concept,bg_new)
                caption_raw="a picture of {} with {} in the background".format(train_prior_concept,bg)
                caption_simple="a picture of {}".format(bg_new)
                caption_simple_raw="a picture of {}".format(bg)
            else:

                caption="a picture of {} with {} in the background".format(self.placeholder,bg_new)
                caption_raw="a picture of {} with {} in the background".format(self.placeholder,bg)
                caption_simple="a picture of {}".format(bg_new)
                caption_simple_raw="a picture of {}".format(bg)
            print(caption_simple,'caption_simple')
            self.captions.append(caption)
            self.captions_raw.append(caption_raw)

            self.captions_simple.append(caption_simple)
            self.captions_simple_raw.append(caption_simple_raw)


        print('num captions:\t{}'.format(len(self.captions)))
        self._length=len(self.captions)
        
        self.get_images = get_images
        self.mask_token_ids = mask_token_ids
        self.mask_prob = mask_prob
        self.mlm_target = mlm_target
        self.exclude_suffix = exclude_suffix
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.train_prior_concept = train_prior_concept
        self.placeholder_token = placeholder_token
        self.size = size
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]
        self.num_images = len(self.image_paths)
        # self._length = self.num_images * repeats
        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        # 1. caption
        caption=self.captions[index%len(self.captions)]
        caption_raw=self.captions_raw[index%len(self.captions)]
        caption=caption.strip()
        caption_raw=caption_raw.strip()


        caption_simple=self.captions_simple[index%len(self.captions)]
        caption_simple_raw=self.captions_simple_raw[index%len(self.captions)]
        caption_simple=caption_simple.strip()
        caption_simple_raw=caption_simple_raw.strip()

        # if self.include_prior_concept:
        #     placeholder='{} {}'.format(self.placeholder_token,self.train_prior_concept)
        # else:
        #     placeholder='{}'.format(self.placeholder_token)
        # if self.prior_only:
        #     caption=caption.replace('<new1>',self.train_prior_concept) # caption without masked embedding
        #     caption=caption.replace('  ',' ')
        # else:
        #     caption=caption.replace('<new1>','{}'.format(placeholder)) # caption without masked embedding
        caption_simple_concept="a picture of {}".format(self.placeholder)
            
        example["input_ids"] = self.tokenizer(
                caption_raw,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]
        example["input_ids_simple_concept"] = self.tokenizer(
                caption_simple_concept,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]
        
        example["raw_caption"]=caption_raw


        words=caption.split()
        is_bg_tokens=[False]       # first token for <startoftext>
        is_keyword_tokens1=[False] # first token for <startoftext>
        masked_idxs=[False] # first token for <startoftext>
        non_special_idxs=[False]   # first token for <startoftext>
        non_keyword_idxs=[True]    # first token for <startoftext>
        is_prior1=[False]
        num_total_tokens=0
        for word_idx in range(len(words)):
            cap_word=words[word_idx]
            cap_word_raw=cap_word.replace('<bg>','')
            word_token_ids=self.tokenizer.encode(cap_word_raw,add_special_tokens=False)
            num_tokens=len(word_token_ids)
            non_special_idxs+=([True]*num_tokens)
            if "<bg>" in cap_word:
                is_bg_tokens=is_bg_tokens+(([True])+([False]*(num_tokens-1)))
            else:
                is_bg_tokens=is_bg_tokens+([False]*num_tokens)
            for tok_id in word_token_ids:
                tok_decoded=self.tokenizer.decode(tok_id)
                # if "<bg>" in cap_word:
                #     is_bg_tokens.append(True)
                # else:
                #     is_bg_tokens.append(False)
                    
                    
                
                if cap_word==self.train_prior_concept:
                    is_prior1.append(True)
                else:
                    is_prior1.append(False)


                
                if tok_decoded==self.placeholder_token:
                    is_keyword_tokens1.append(True)
                    non_keyword_idxs.append(False)
                else:
                    is_keyword_tokens1.append(False)
                    non_keyword_idxs.append(True)
        # print(is_bg_tokens,'is_bg_tokens')
        # exit(0)
        # is_keyword_tokens1
        for _ in range(len(is_keyword_tokens1),self.tokenizer.model_max_length):
            is_keyword_tokens1.append(False)
        assert len(is_keyword_tokens1)==self.tokenizer.model_max_length
        for _ in range(len(is_bg_tokens),self.tokenizer.model_max_length):
            is_bg_tokens.append(False)
        assert sum(is_bg_tokens)==1
        assert len(is_bg_tokens)==self.tokenizer.model_max_length
        if not self.prior_only:
            assert sum(is_keyword_tokens1)==1
        else:
            assert sum(is_keyword_tokens1)==0
        example["is_keyword_tokens1"]=torch.BoolTensor(is_keyword_tokens1)


        # non_keyword_idxs
        for _ in range(len(non_keyword_idxs),self.tokenizer.model_max_length):
            non_keyword_idxs.append(True)
        non_keyword_idxs=torch.BoolTensor(non_keyword_idxs)
        assert len(non_keyword_idxs)==self.tokenizer.model_max_length
        if not self.prior_only: # keyword==1
            assert torch.sum(non_keyword_idxs)==(self.tokenizer.model_max_length-1),'torch.sum(non_keyword_idxs)==(self.tokenizer.model_max_length-1)'
        else:
            assert torch.sum(non_keyword_idxs)==(self.tokenizer.model_max_length),'torch.sum(non_keyword_idxs)==(self.tokenizer.model_max_length)'
        example['non_keyword_idxs']=non_keyword_idxs

        
        
        # is_prior1
        for _ in range(len(is_prior1),self.tokenizer.model_max_length):
            is_prior1.append(False)
        is_prior1=torch.BoolTensor(is_prior1)
        assert len(is_prior1)==self.tokenizer.model_max_length
        if self.include_prior_concept:
            assert torch.sum(is_prior1)==1,'torch.sum(is_prior1)==1'
        example['is_prior1']=is_prior1

        

        
        # non_special_idxs
        non_special_idxs=non_special_idxs[:self.tokenizer.model_max_length-1]
        for _ in range(len(non_special_idxs),self.tokenizer.model_max_length):
            non_special_idxs.append(False)
        non_special_idxs=torch.BoolTensor(non_special_idxs)
        example['non_special_idxs']=non_special_idxs


        # is_bg_tokens
        for _ in range(len(is_bg_tokens),self.tokenizer.model_max_length):
            is_bg_tokens.append(False)
        assert len(is_bg_tokens)==self.tokenizer.model_max_length
        example["is_bg_tokens"]=torch.BoolTensor(is_bg_tokens)


        # simple bg
        example["input_ids_simple"] = self.tokenizer(
                caption_simple_raw,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]
        example["raw_caption_simple"]=caption_simple_raw


        
        words_simple_bg=caption_simple.split()
        is_bg_tokens_simple=[False]      # first token for <startoftext>
        is_keyword_tokens1_simple=[False]       # first token for <startoftext>
        for word_idx in range(len(words_simple_bg)):
            cap_word_simple_bg=words_simple_bg[word_idx]
            cap_word_simple_bg_raw=cap_word_simple_bg.replace('<bg>','')
            word_token_ids_simple_bg=self.tokenizer.encode(cap_word_simple_bg_raw,add_special_tokens=False)
            num_tokens=len(word_token_ids)
            if "<bg>" in cap_word_simple_bg:
                is_bg_tokens_simple=is_bg_tokens_simple+(([True])+([False]*(num_tokens-1)))
            else:
                is_bg_tokens_simple=is_bg_tokens_simple+([False]*num_tokens)
            for tok_id in word_token_ids_simple_bg:
                tok_decoded=self.tokenizer.decode(tok_id)
                if tok_decoded==self.placeholder_token:
                    is_keyword_tokens1_simple.append(True)
                else:
                    is_keyword_tokens1_simple.append(False)
        # print()
        # is_bg_tokens_simple
        for _ in range(len(is_bg_tokens_simple),self.tokenizer.model_max_length):
            is_bg_tokens_simple.append(False)
        assert len(is_bg_tokens_simple)==self.tokenizer.model_max_length
        assert sum(is_bg_tokens_simple)==1
        example["is_bg_tokens_simple"]=torch.BoolTensor(is_bg_tokens_simple)   

        # is_keyword_tokens1_simple
        for _ in range(len(is_keyword_tokens1_simple),self.tokenizer.model_max_length):
            is_keyword_tokens1_simple.append(False)
        assert len(is_keyword_tokens1_simple)==self.tokenizer.model_max_length
        assert sum(is_keyword_tokens1_simple)==0
        example["is_keyword_tokens1_simple"]=torch.BoolTensor(is_keyword_tokens1_simple)
                    
            

        

        return example

