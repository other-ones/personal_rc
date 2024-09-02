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
    "a rendering of {}",
    "a cropped photo of the {}",
    "the photo of {}",
    "a photo of clean {}",
    "a photo of dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of {}",
    "a bright photo of the {}",
    "a cropped photo of {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of {}",
    "a photo of nice {}",
    "a good photo of {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of cool {}",
    "a photo of small {}",
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

class TextualInversionDatasetSingle(Dataset):
    def __init__(
        self,
        data_root1,
        tokenizer,
        include_prior_concept,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0,
        center_crop=False,
        exclude_suffix=True,
        mlm_target='all',
        mask_prob=0.25,
        mask_token_ids=None,
        get_images=True,
        prompt_type=None,
        placeholder_token1=None,
        placeholder_id1=None,
        prior_concept1=None,
        caption_root=None,

    ):
        self.include_prior_concept=include_prior_concept
        self.placeholder_id1 = placeholder_id1
        self.placeholder_token1 = placeholder_token1
        self.prior_concept1 = prior_concept1
        caption_dir_path=os.path.join(caption_root,prompt_type)
        self.prompt_type=prompt_type
        self.captions={}
        max_length=0
        cap_file_list=os.listdir(caption_dir_path)
        for cap_file in cap_file_list:
            fname=cap_file.split('.')[0]
            cap_file_path=os.path.join(caption_dir_path,cap_file)
            self.captions[fname]=open(cap_file_path).readlines()
            print('{}\t{}'.format(fname,len(self.captions[fname])))
            if max_length<len(self.captions[fname]):
                max_length=len(self.captions[fname])
        self._length=max_length
        self.caption_types=list(self.captions.keys())
        
        self.get_images = get_images
        self.mask_token_ids = mask_token_ids
        self.mask_prob = mask_prob
        self.mlm_target = mlm_target
        self.exclude_suffix = exclude_suffix
        self.data_root1 = data_root1
        self.tokenizer = tokenizer
        # self.prior_concept_id=tokenizer.encode(self.prior_concept1,add_special_tokens=False)[0]
        self.learnable_property = learnable_property
        self.size = size
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.image_paths1 = [os.path.join(self.data_root1, file_path) for file_path in os.listdir(self.data_root1)]
        self.image_paths=[self.image_paths1]
        self.num_images = len(self.image_paths1)
        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        
    def __len__(self):
        return self._length

    
    def __getitem__(self, index):
        example = {}
        sampled_type=np.random.choice(self.caption_types)
        sampled_caption=self.captions[sampled_type][index%len(self.captions[sampled_type])]
        # 3. MLM
        if self.include_prior_concept:
            placeholder1='{} {}'.format(self.placeholder_token1,self.prior_concept1)
        else:
            placeholder1='{}'.format(self.placeholder_token1)
        
        sampled_caption=sampled_caption.replace('<new1>',placeholder1)
        example['raw_caption']=sampled_caption
        input_ids_pos = self.tokenizer(
                sampled_caption,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]
        example['input_ids_pos']=input_ids_pos
        

        
        
        
        is_prior1=[False]
        is_keyword_tokens1=[False] # first token for <startoftext>
        words_split=sampled_caption.split()
        non_special_idxs=[False]
        non_keyword_idxs=[False]
        for word_idx in range(len(words_split)):
            cap_word=words_split[word_idx]
            word_token_ids=self.tokenizer.encode(cap_word,add_special_tokens=False)
            num_tokens=len(word_token_ids)
            non_special_idxs+=([True]*num_tokens)
            for tok_id in word_token_ids:
                tok_decoded=self.tokenizer.decode(tok_id)
                # 2) keyword indices and labels for MLM.
                if tok_id in self.placeholder_id1:
                    if tok_id == self.placeholder_id1[0]:
                        assert num_tokens==1
                        is_keyword_tokens1.append(True)
                    else:
                        is_keyword_tokens1.append(False)
                    non_keyword_idxs.append(False)
                else:
                    non_keyword_idxs.append(True)
                    is_keyword_tokens1.append(False)
                # prior1
                if tok_decoded==self.prior_concept1:
                    is_prior1.append(True)
                else:
                    is_prior1.append(False)
       
        

        
        

        # prior1
        for _ in range(len(is_prior1),self.tokenizer.model_max_length):
            is_prior1.append(False)
        is_prior1=torch.BoolTensor(is_prior1)
        assert len(is_prior1)==self.tokenizer.model_max_length
        assert torch.sum(is_prior1)==1,'torch.sum(is_prior1)==1'
        example['is_prior1']=is_prior1
        


        # non_special_idxs
        non_special_idxs=non_special_idxs[:self.tokenizer.model_max_length-1]
        for _ in range(len(non_special_idxs),self.tokenizer.model_max_length):
            non_special_idxs.append(False)
        non_special_idxs=torch.BoolTensor(non_special_idxs)
        example['non_special_idxs']=non_special_idxs
        # non_special_idxs

        
        
        # non_keyword_idxs
        non_keyword_idxs=non_keyword_idxs[:self.tokenizer.model_max_length-1]
        for _ in range(len(non_keyword_idxs),self.tokenizer.model_max_length):
            non_keyword_idxs.append(True)
        non_keyword_idxs=torch.BoolTensor(non_keyword_idxs)
        example['non_keyword_idxs']=non_keyword_idxs
        # non_keyword_idxs
        
        # is_keyword_tokens_mlm
        for _ in range(len(is_keyword_tokens1),self.tokenizer.model_max_length):
            is_keyword_tokens1.append(False)
        assert len(is_keyword_tokens1)==self.tokenizer.model_max_length
        is_keyword_tokens1=torch.BoolTensor(is_keyword_tokens1)
        example['is_keyword_tokens1']=is_keyword_tokens1
        # is_keyword_tokens_mlm

        
        
        return example

