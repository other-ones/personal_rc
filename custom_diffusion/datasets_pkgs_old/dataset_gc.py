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
from PIL import Image,ImageDraw
import string
import albumentations as A
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


class GCDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """
    def __init__(
            self,
            caption_path='generated/captions_filtered.txt',
            data_root='generated/samples',
            tokenizer=None,
            placeholder_token=None,
    ):
        self.data_root=data_root
        self.placeholder_token=placeholder_token
        self.tokenizer=tokenizer
        self.fnames=[]
        self.captions=[]
        self.nouns=[]
        caption_lines=open(caption_path).readlines()

        for line in caption_lines:
            fname,caption,noun=line.strip().split('\t')
            self.fnames.append(fname)
            self.captions.append(caption)
            self.nouns.append(noun)
        
        self.num_instance_images=len(caption_lines)
        self.random_trans = A.Compose([
            A.Resize(height=224, width=224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20),
            A.Blur(p=0.3),
            A.ElasticTransform(p=0.3)
        ])
    def get_tensor_clip(self, normalize=True, toTensor=True):
        transform_list = []
        if toTensor:
            transform_list += [torchvision.transforms.ToTensor()]
        if normalize:
            transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                                (0.26862954, 0.26130258, 0.27577711))]
        return torchvision.transforms.Compose(transform_list)
    def __len__(self):
        return self.num_instance_images#len(self.db_list)
    def __getitem__(self, index):
        example = {}
        noun=self.nouns[index]
        fname=self.fnames[index]
        caption_wo_keyword=self.captions[index]
        caption=caption_wo_keyword.replace(self.placeholder_token,noun)
        img_path=os.path.join(self.data_root,fname)
        image = Image.open(img_path).resize((512,512))
        if not image.mode == "RGB":
            image = image.convert("RGB")
        
        # InputImage
        image = image.resize((512, 512), resample=self.interpolation)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        # InputImage

        
        
        # Caption
        example["input_ids"] = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        example['caption']=caption
        # Caption

        # is_keyword_token
        caption_wo_keyword_splits=caption_wo_keyword.split()
        is_keyword_tokens=[False] # first token for <startoftext>
        for cap_idx in range(len(caption_wo_keyword_splits)):
            cap_word=caption_wo_keyword_splits[cap_idx]
            word_token_ids=self.tokenizer.encode(cap_word,add_special_tokens=False)
            num_tokens=len(word_token_ids)
            for tidx in range(num_tokens):
                if self.placeholder_token in cap_word:
                    assert num_tokens==1
                    is_keyword_tokens.append(True)
                else:
                    is_keyword_tokens.append(False)
        # padding
        for _ in range(len(is_keyword_tokens),self.tokenizer.model_max_length):
            is_keyword_tokens.append(False)
        assert len(is_keyword_tokens)==self.tokenizer.model_max_length
        is_keyword_tokens=torch.BoolTensor(is_keyword_tokens)
        example["is_keyword_tokens"]=torch.BoolTensor(is_keyword_tokens)
        # is_keyword_token

        # CLIPImage
        img_p_np = cv2.imread(img_path)
        img_p_np = cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)
        image_tensor = img_p_np#[bbox_pad[0]:bbox_pad[1], bbox_pad[2]:bbox_pad[3], :]
        ref_image_tensor = self.random_trans(image=image_tensor)
        ref_image_tensor = Image.fromarray(ref_image_tensor["image"])
        ref_image_tensor=self.get_tensor_clip()(ref_image_tensor)
        print(ref_image_tensor.shape,'ref_image_tensor.shape')
        example["pixel_values_clip"] = ref_image_tensor
        # CLIPImage


        example['noun']=noun
        example['fname']=fname
        return example

if __name__=='__main__':
    gc_dataset=GCDataset()
    len_dataset=len(gc_dataset)
    print(len_dataset)
    for idx in range(len_dataset):
        example=gc_dataset.__getitem__(idx)
        print(example.keys())
