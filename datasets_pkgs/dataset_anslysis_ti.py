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
        prior_concept=None,
        placeholder_token=None,
        caption_root=None,
    ):
        self.prompt_type=prompt_type
        self.include_prior_concept=include_prior_concept
        caption_dir_path=os.path.join(caption_root,prompt_type)
        cap_file_list=os.listdir(caption_dir_path)
        self.captions={}
        max_length=0
        for cap_file in cap_file_list:
            fname=cap_file.split('.')[0]
            if prompt_type is not None and prompt_type not in fname:
                continue
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
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.prior_concept = prior_concept
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
        sampled_type=np.random.choice(self.caption_types)
        caption=self.captions[sampled_type][index%len(self.captions[sampled_type])]
        if self.include_prior_concept:
            placeholder='{} {}'.format(self.placeholder_token,self.prior_concept)
        else:
            placeholder='{}'.format(self.placeholder_token)
        caption=caption.replace('<new1>','{}'.format(placeholder)) # caption without masked embedding
        example["input_ids_pos"] = self.tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]
        example["raw_caption"]=caption


        words=caption.split()
        is_keyword_tokens1=[False] # first token for <startoftext>
        non_special_idxs=[False]
        non_keyword_idxs=[False]
        is_prior1=[False]
        for word_idx in range(len(words)):
            cap_word=words[word_idx]
            word_token_ids=self.tokenizer.encode(cap_word,add_special_tokens=False)
            num_tokens=len(word_token_ids)
            non_special_idxs+=([True]*num_tokens)
            for tok_id in word_token_ids:
                # 2) keyword indices and labels for MLM
                tok_decoded=self.tokenizer.decode(tok_id)
                if self.placeholder_token == cap_word:
                    assert num_tokens==1
                    is_keyword_tokens1.append(True)
                    non_keyword_idxs.append(False)
                else:
                    is_keyword_tokens1.append(False)
                    non_keyword_idxs.append(True)
                if tok_decoded==self.prior_concepts[0]:
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


        # 3) is_keyword_tokens_mlm - keyword indices for MLM
        for _ in range(len(is_keyword_tokens1),self.tokenizer.model_max_length):
            is_keyword_tokens1.append(False)
        assert len(is_keyword_tokens1)==self.tokenizer.model_max_length
        assert sum(is_keyword_tokens1)==1
        example["is_keyword_tokens1"]=torch.BoolTensor(is_keyword_tokens1)

        

        
        
        # 6) non_special_idxs/masked_idxs
        non_special_idxs=non_special_idxs[:self.tokenizer.model_max_length-1]
        for _ in range(len(non_special_idxs),self.tokenizer.model_max_length):
            non_special_idxs.append(False)
        non_special_idxs=torch.BoolTensor(non_special_idxs)
        example['non_special_idxs']=non_special_idxs

        return example

