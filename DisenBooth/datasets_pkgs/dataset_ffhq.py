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

imagenet_templates_small = [
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
class FFHQDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """
    def __init__(
            self,
            mlm_caption_path='/data/dataset/coco/karpathy/coco_person_caption.txt',
            data_root='/data/dataset/ffhd/images/images1024x1024',
            tokenizer=None,
            placeholder_token=None,
            interpolation="bicubic",
            num_vectors=5,
            fixed_target=None,
            mlm_target='masked',
            mask_prob=0.15,
            whole_word_mask=False,
            mask_token_ids=None,
    ):
        self.whole_word_mask=whole_word_mask
        self.mask_token_ids=mask_token_ids
        self.mask_prob=mask_prob
        self.mlm_target=mlm_target
        self.fixed_target=fixed_target
        self.num_vectors=num_vectors
        self.data_root=data_root
        self.placeholder_token=placeholder_token
        self.tokenizer=tokenizer
        self.fnames=[]
        self.mlm_captions=[]
        self.class_concepts=[]
        mlm_caption_lines=open(mlm_caption_path).readlines()
        for line in mlm_caption_lines:
            caption,class_concept=line.strip().split('\t')
            self.mlm_captions.append(caption)
            self.class_concepts.append(class_concept)

        flist=os.listdir(data_root)
        for fname in flist:
            self.fnames.append(fname)
        self.class_concepts=np.array(self.class_concepts)
        self.num_instance_images=len(self.fnames)
        self.random_trans = A.Compose([
            A.Resize(height=224, width=224),
        ])
        self.fnames=np.array(self.fnames)
        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]
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
        samples_idxs=np.random.choice(np.arange(len(self.fnames)),size=3,replace=False)
        sampled_fnames=self.fnames[samples_idxs]
        sampled_templates=np.random.choice(imagenet_templates_small,size=3,replace=False)
        sampled_images=[]
        sampled_captions=[]
        
        for fidx,fname in enumerate(sampled_fnames):
            img_path=os.path.join(self.data_root,fname)
            image = Image.open(img_path).resize((512,512))
            if not image.mode == "RGB":
                image = image.convert("RGB")
            image = image.resize((512, 512), resample=self.interpolation)
            image = np.array(image).astype(np.uint8)
            image = (image / 127.5 - 1.0).astype(np.float32)
            sampled_images.append(image)
            temp=sampled_templates[fidx]
            caption=temp.format(self.placeholder_token*self.num_vectors)
            sampled_captions.append(caption)


        if self.fixed_target:
            img_path=self.fixed_target
            image = Image.open(img_path).resize((512,512))
            if not image.mode == "RGB":
                image = image.convert("RGB")
            image = image.resize((512, 512), resample=self.interpolation)
            image = np.array(image).astype(np.uint8)
            image = (image / 127.5 - 1.0).astype(np.float32)
            sampled_images[0]=image
        
        # 1. InputImage / RefImage(for reg)
        image_target=sampled_images[0]
        image_ref1=sampled_images[1] 
        image_ref2=sampled_images[2]
        example["pixel_values_target"] = torch.from_numpy(image_target).permute(2, 0, 1)
        example["pixel_values_ref1"] = torch.from_numpy(image_ref1).permute(2, 0, 1)
        example["pixel_values_ref2"] = torch.from_numpy(image_ref2).permute(2, 0, 1)
        # 1. InputImage



        # 2. input_ids
        example["input_ids_target"] = self.tokenizer(
            sampled_captions[0],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        example["input_ids_ref1"] = self.tokenizer(
            sampled_captions[1],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        example["input_ids_ref2"] = self.tokenizer(
            sampled_captions[2],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        # 2. input_ids

        
        # 3. is_keyword_token
        for cidx,caption in enumerate(sampled_captions):
            caption_splits=caption.split()
            is_keyword_tokens=[False] # first token for <startoftext>
            for cap_idx in range(len(caption_splits)):
                cap_word=caption_splits[cap_idx]
                word_token_ids=self.tokenizer.encode(cap_word,add_special_tokens=False)
                num_tokens=len(word_token_ids)
                for tidx in range(num_tokens):
                    if self.placeholder_token in cap_word:
                        is_keyword_tokens.append(True)
                    else:
                        is_keyword_tokens.append(False)
            for _ in range(len(is_keyword_tokens),self.tokenizer.model_max_length):
                is_keyword_tokens.append(False)
            assert len(is_keyword_tokens)==self.tokenizer.model_max_length
            is_keyword_tokens=torch.BoolTensor(is_keyword_tokens)
            if cidx==0:
                example["is_keyword_tokens_target"]=torch.BoolTensor(is_keyword_tokens)
            else:
                example["is_keyword_tokens_ref{}".format(cidx)]=torch.BoolTensor(is_keyword_tokens)
        # 3. is_keyword_token

        # 4. CLIPImage
        for fidx,fname in enumerate(sampled_fnames):
            img_path=os.path.join(self.data_root,fname)
            img_p_np = cv2.imread(img_path)
            img_p_np = cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)
            image_pil_224 = Image.fromarray(img_p_np).resize((224,224))
            image_tensor=self.get_tensor_clip()(image_pil_224)
            if fidx==0:
                example["pixel_values_clip_target"] = image_tensor
                example['fname_target']=fname
            else:
                example["pixel_values_clip_ref{}".format(fidx)] = image_tensor
                example['fname_ref{}'.format(fidx)]=fname
        # 4. CLIPImage


        # 5. MLM
        # mlm_caption=np.random.choice(self.mlm_captions)
        sampled_mlm_caption=np.random.choice(self.mlm_captions)
        sampled_mlm_caption=sampled_mlm_caption.replace('[KEYWORD]',self.placeholder_token*self.num_vectors)
        mlm_words=sampled_mlm_caption.split()
        is_keyword_tokens_mlm=[False] # first token for <startoftext>

        masked_idxs=[False]
        if self.mlm_target=='non_special' or self.mlm_target =='masked': # if non-special only then bos is not learned
            mlm_labels=[-100]
        else:
            mlm_labels=[self.tokenizer.bos_token_id]
        input_ids_masked=[self.tokenizer.bos_token_id]
        non_special_idxs=[False]
        for word_idx in range(len(mlm_words)):
            masked=False
            if np.random.rand()<self.mask_prob: #15% of prob -> do whole word masking
                masked=True
            cap_word=mlm_words[word_idx]
            word_token_ids=self.tokenizer.encode(cap_word,add_special_tokens=False)
            num_tokens=len(word_token_ids)
            non_special_idxs+=([True]*num_tokens)
            for tok_id in word_token_ids:
                if not self.whole_word_mask:
                    if np.random.rand()<self.mask_prob and self.placeholder_token not in cap_word: 
                        masked=True
                    else:
                        masked=False
                if self.placeholder_token in cap_word:
                    is_keyword_tokens_mlm.append(True)
                else:
                    is_keyword_tokens_mlm.append(False)
                # 1. mlm_labels
                if self.mlm_target=='masked':
                    if masked:
                        mlm_labels.append(tok_id)
                    else:
                        mlm_labels.append(-100)
                elif self.mlm_target=='non_special' or self.mlm_target=='all': #append regardless of masked
                    mlm_labels.append(tok_id)

                # 2. masked_idxs
                if masked:
                    masked_idxs.append(True)
                    input_ids_masked.append(self.mask_token_ids)
                else:
                    masked_idxs.append(False)
                    input_ids_masked.append(tok_id)
        for _ in range(len(is_keyword_tokens_mlm),self.tokenizer.model_max_length):
            is_keyword_tokens_mlm.append(False)
        assert len(is_keyword_tokens_mlm)==self.tokenizer.model_max_length
        is_keyword_tokens_mlm=torch.BoolTensor(is_keyword_tokens_mlm)
        example["is_keyword_tokens_mlm"]=torch.BoolTensor(is_keyword_tokens)
        input_ids_masked=input_ids_masked[:self.tokenizer.model_max_length-1]
        input_ids_masked.append(self.tokenizer.eos_token_id)
        for _ in range(len(input_ids_masked),self.tokenizer.model_max_length):
            input_ids_masked.append(self.tokenizer.pad_token_id)
        input_ids_masked=torch.LongTensor(input_ids_masked)
        example["input_ids_masked"]=input_ids_masked
        masked_idxs=masked_idxs[:self.tokenizer.model_max_length-1]
        mlm_labels=mlm_labels[:self.tokenizer.model_max_length-1]
        non_special_idxs=non_special_idxs[:self.tokenizer.model_max_length-1]
        if self.mlm_target not in ['non_special','masked']: #if all learned (bos+eos+nonspecial) then add eos token
            mlm_labels.append(self.tokenizer.eos_token_id)
        # We don't learn pad tokens 
        for _ in range(len(mlm_labels),self.tokenizer.model_max_length):
            mlm_labels.append(-100)
        for _ in range(len(masked_idxs),self.tokenizer.model_max_length):
            masked_idxs.append(False)
            non_special_idxs.append(False)
        assert len(masked_idxs)==self.tokenizer.model_max_length
        assert len(mlm_labels)==self.tokenizer.model_max_length
        masked_idxs=torch.BoolTensor(masked_idxs)
        mlm_labels=torch.LongTensor(mlm_labels)
        non_special_idxs=torch.BoolTensor(non_special_idxs)
        example['masked_idxs']=masked_idxs
        example['mlm_labels']=mlm_labels
        example['non_special_idxs']=non_special_idxs
        input_ids_non_mask= self.tokenizer(
            sampled_mlm_caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        example["input_ids_non_mask"]=input_ids_non_mask
        # 5. MLM
        return example

if __name__=='__main__':
    gc_dataset=FFHQDataset()
    len_dataset=len(gc_dataset)
    print(len_dataset)
    for idx in range(len_dataset):
        example=gc_dataset.__getitem__(idx)
        print(example.keys())
