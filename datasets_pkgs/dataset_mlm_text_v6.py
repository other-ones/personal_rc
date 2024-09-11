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

mlm_prefixes=sorted(list(set([
    "a photo of {}",
    "a rendering of {}",
    "a cropped photo of {}",
    "the photo of {}",
    "a photo of my {}",
    "a close-up photo of {}",
    "a cropped photo of {}",
    "a close-up photo of {}",
    "a rendition of {}",
])))
imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of {}",
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
class MLMDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """
    def __init__(
            self,
            real_caption_path='/data/dataset/coco/karpathy/coco_caption_raw.txt',
            synth_caption_root='../datasets_pkgs/captions/contextnet/all',
            tokenizer=None,
            mask_tokens=None,
            mask_token_ids=None,
            mask_prob=0.15,
            min_length=5,
            mlm_target='non_special',
            whole_word_mask=True,
            synth_prob=0.3
    ):
        self.synth_prob=synth_prob
        self.mask_token_ids=mask_token_ids
        self.whole_word_mask=whole_word_mask
        self.mlm_target=mlm_target
        self.mask_prob=mask_prob
        self.mask_tokens=mask_tokens
        self.tokenizer=tokenizer
        self.real_captions=[]
        # self.captions=open(caption_path).readlines()
        self.num_instance=0
        # cap_types=os.listdir(aug_root)
        # self.aug_captions={}
        self.synth_captions={}
        for cap_file in os.listdir(synth_caption_root):
            cap_name=cap_file.split('.')[0]
            lines=list(set(open(os.path.join(synth_caption_root,cap_file)).readlines()))
            self.synth_captions[cap_name]=lines
            self.num_instance+=len(lines)
        self.synth_cap_types=list(self.synth_captions.keys())
        caption_lines=list(set(open(real_caption_path).readlines()))
        for line in caption_lines:
            caption=line.strip()
            words=caption.split()
            # if len(words)<min_length:
            #     continue
            self.real_captions.append(caption)
        self.real_captions=np.array(self.real_captions)
        self.num_instance+=len(self.captions)
        print(self.num_instance,'num_instance')
        
        
    
    def __len__(self):
        return self.num_instance#len(self.db_list)
    def __getitem__(self, index):
        example = {}
        if np.random.rand()<(1-self.synth_prob):
            sampled_caption=self.real_captions[index%(len(self.real_captions))]
        else:
            sampled_type=np.random.choice(self.synth_cap_types)
            sampled_caption=self.synth_captions[sampled_type][index%(len(self.synth_captions[sampled_type]))]
            prefix=np.random.choice(mlm_prefixes)
            sampled_caption=prefix.format(sampled_caption)


        example['raw_caption']=sampled_caption
        # 2. input_ids
        input_ids= self.tokenizer(
            sampled_caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        sampled_words=sampled_caption.split()
        example["input_ids"]=input_ids
        # 2. input_ids
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        # 3. mask_idxs
        masked_idxs=[False]
        # mlm_target: 
        # non_special: non_special toke nonly
        # all: bos+eos+non_special+masked
        # we don't learn pading
        if self.mlm_target in ['non_special','masked']: # if non-special only then bos is not learned
            mlm_labels=[-100]
        elif self.mlm_target in ['all','non_padding']:
            mlm_labels=[self.tokenizer.bos_token_id]
        else:
            assert False





        input_ids_masked=[self.tokenizer.bos_token_id]
        non_special_idxs=[False]
        # non specials
        for word_idx in range(len(sampled_words)):
            masked=False
            if np.random.rand()<self.mask_prob: #15% of prob -> do whole word masking
                masked=True
            cap_word=sampled_words[word_idx]
            word_token_ids=self.tokenizer.encode(cap_word,add_special_tokens=False)
            num_tokens=len(word_token_ids)
            non_special_idxs+=([True]*num_tokens)
            for tok_id in word_token_ids:
                # if whole_word_mask=False
                # then sample the mask probability again
                # otherwise use the previous result
                if not self.whole_word_mask:
                    if np.random.rand()<self.mask_prob: 
                        masked=True
                    else:
                        masked=False




                # 1. mlm_labels
                # if mlm_target=='masked' 
                # then append the token id
                if self.mlm_target=='masked':
                    if masked:
                        mlm_labels.append(tok_id)
                    else:
                        mlm_labels.append(-100)

                # if mlm_target !='masked'
                # append regardless of masked
                elif self.mlm_target in ['non_special','all','non_padding']: 
                    mlm_labels.append(tok_id)
                else:
                    assert False
                # 1. mlm_labels


                # 2. masked_idxs
                if masked:
                    masked_idxs.append(True)
                    input_ids_masked.append(self.mask_token_ids)
                else:
                    masked_idxs.append(False)
                    input_ids_masked.append(tok_id)
                




        # input_ids_masked
        input_ids_masked=input_ids_masked[:self.tokenizer.model_max_length-1]
        input_ids_masked.append(self.tokenizer.eos_token_id)
        for _ in range(len(input_ids_masked),self.tokenizer.model_max_length):
            input_ids_masked.append(self.tokenizer.pad_token_id)
        input_ids_masked=torch.LongTensor(input_ids_masked)
        example["input_ids_masked"]=input_ids_masked
        # masked_idxs
        masked_idxs=masked_idxs[:self.tokenizer.model_max_length-1]
        # masked_labels
        mlm_labels=mlm_labels[:self.tokenizer.model_max_length-1]
        # non_special_idxs
        non_special_idxs=non_special_idxs[:self.tokenizer.model_max_length-1]


        


        # if self.mlm_target not in ['non_special','masked']: #if all learned (bos+eos+nonspecial) then add eos token
        if self.mlm_target in ['all','non_padding']:
            mlm_labels.append(self.tokenizer.eos_token_id)
        elif self.mlm_target in ['masked','non_special']: #masked/non_special
            mlm_labels.append(-100)
        else:
            assert False
        




        # We don't learn pad tokens 
        for _ in range(len(mlm_labels),self.tokenizer.model_max_length):
            if self.mlm_target=='all':
                mlm_labels.append(self.tokenizer.pad_token_id)
            else: # non_padding/non_special/masked
                mlm_labels.append(-100)
            # mlm_labels.append(-100)
        for _ in range(len(masked_idxs),self.tokenizer.model_max_length):
            masked_idxs.append(False)
            non_special_idxs.append(False)
        # decoded_masked=self.tokenizer.batch_decode(input_ids_masked[masked_idxs])
        # print(decoded_masked,'decoded_masked')
        # masked_ids=input_ids_masked[masked_idxs]
        # label_ids=input_ids[masked_idxs]
        # non_specials_input=input_ids[non_special_idxs]
        # non_specials_masked=input_ids_masked[non_special_idxs]
        # decoded_masked=self.tokenizer.batch_decode(masked_ids)
        # decoded_labels=self.tokenizer.batch_decode(label_ids)
        # non_special_decoded=self.tokenizer.batch_decode(non_specials_input)
        # non_special_masked_decoded=self.tokenizer.batch_decode(non_specials_masked)
        # print(non_special_decoded,'non_special_decoded')
        # print(non_special_masked_decoded,'non_special_masked_decoded')
        # print(decoded_masked,'decoded_masked',np.sum(masked_idxs))
        # print(decoded_labels,'decoded_labels',np.sum(masked_idxs))
        # non_mask_idxs=[not item for item in masked_idxs]
        # exit()

        assert len(masked_idxs)==self.tokenizer.model_max_length
        assert len(mlm_labels)==self.tokenizer.model_max_length
        masked_idxs=torch.BoolTensor(masked_idxs)
        mlm_labels=torch.LongTensor(mlm_labels)
        # assert torch.all(mlm_labels[non_mask_idxs]==(-100)).item()
        # print('all -100 passed')
        non_special_idxs=torch.BoolTensor(non_special_idxs)
        example['masked_idxs']=masked_idxs
        example['mlm_labels']=mlm_labels
        example['non_special_idxs']=non_special_idxs
        # 3. mask_idxs
        
        return example

if __name__=='__main__':
    gc_dataset=MLMDataset()
    len_dataset=len(gc_dataset)
    print(len_dataset)
    for idx in range(len_dataset):
        example=gc_dataset.__getitem__(idx)
        print(example.keys())
