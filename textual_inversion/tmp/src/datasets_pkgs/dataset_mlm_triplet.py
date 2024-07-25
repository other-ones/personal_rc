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
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0,
        placeholder_token="*",
        center_crop=False,
        exclude_suffix=True,
        prior_concept=None,
        mlm_target='all',
        mask_prob=0.15,
        mask_token_ids=None,
        learn_prior=False,
        get_images=True,
        prompt_type=None
    ):
        self.prompt_type=prompt_type
        self.include_prior_concept=include_prior_concept
        if prompt_type=='nonliving':
            from .mlm_pkgs.caption_generator_nonliving import CaptionGeneratorNonLiving
            self.prompt_generator=CaptionGeneratorNonLiving()
        elif prompt_type=='pet':
            from .mlm_pkgs.caption_generator_pet import CaptionGeneratorPet
            self.prompt_generator=CaptionGeneratorPet()
        elif prompt_type=='building':
            from .mlm_pkgs.caption_generator_building import CaptionGeneratorBuilding
            self.prompt_generator=CaptionGeneratorBuilding()
        else:
            raise Exception('Not Implemented')
        
        self.get_images = get_images
        self.learn_prior = learn_prior
        self.mask_token_ids = mask_token_ids
        self.mask_prob = mask_prob
        self.mlm_target = mlm_target
        self.exclude_suffix = exclude_suffix
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.prior_concept = prior_concept
        # self.prior_concept_id=tokenizer.encode(self.prior_concept,add_special_tokens=False)[0]
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]
        self.num_images = len(self.image_paths)
        self._length = self.num_images * repeats
        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        # self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        # 1. Image
        if self.get_images:
            image = Image.open(self.image_paths[index % (len(self.image_paths))])
            if not image.mode == "RGB":
                image = image.convert("RGB")
            # default to score-sde preprocessing
            img = np.array(image).astype(np.uint8)
            if self.center_crop: #NO
                crop = min(img.shape[0], img.shape[1])
                (
                    h,
                    w,
                ) = (
                    img.shape[0],
                    img.shape[1],
                )
                img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

            image = Image.fromarray(img)
            image = image.resize((self.size, self.size), resample=self.interpolation)
            image = self.flip_transform(image)
            image = np.array(image).astype(np.uint8)
            image = (image / 127.5 - 1.0).astype(np.float32)
            example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)

            # 2. Caption for TI
            placeholder_string = self.placeholder_token
            if self.include_prior_concept:
                text = random.choice(prefixes).format(placeholder_string)+' {}'.format(self.prior_concept)
            else:
                text = random.choice(prefixes).format(placeholder_string)
            example["input_ids"] = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]
            is_keyword_tokens=[False]
            text_words=text.split()
            for word_idx in range(len(text_words)):
                cap_word=text_words[word_idx]
                word_token_ids=self.tokenizer.encode(cap_word,add_special_tokens=False)
                num_tokens=len(word_token_ids)
                for tok_id in word_token_ids:
                    if self.placeholder_token in cap_word:
                        is_keyword_tokens.append(True)
                    else:
                        is_keyword_tokens.append(False)

            # 3) is_keyword_tokens - keyword indices for MLM
            for _ in range(len(is_keyword_tokens),self.tokenizer.model_max_length):
                is_keyword_tokens.append(False)
            assert len(is_keyword_tokens)==self.tokenizer.model_max_length
            example["is_keyword_tokens"]=torch.BoolTensor(is_keyword_tokens)












        # 3. MLM
        mlm_caption=self.prompt_generator.generate_caption(triplet=self.triplet)
        if self.include_prior_concept:
            placeholder='{} {}'.format(self.placeholder_token,self.prior_concept)
        else:
            placeholder='{}'.format(self.placeholder_token)
        mlm_caption=mlm_caption.replace('<new>','{}'.format(placeholder))
        mlm_words=mlm_caption.split()
        is_keyword_tokens_mlm=[False] # first token for <startoftext>
        masked_idxs=[False]
        if self.mlm_target=='non_special': # if non-special only then bos is not learned
            mlm_labels=[-100]
        else:
            mlm_labels=[self.tokenizer.bos_token_id]
        input_ids_masked=[self.tokenizer.bos_token_id]
        non_special_idxs=[False]
        for word_idx in range(len(mlm_words)):
            cap_word=mlm_words[word_idx]
            word_token_ids=self.tokenizer.encode(cap_word,add_special_tokens=False)
            num_tokens=len(word_token_ids)
            non_special_idxs+=([True]*num_tokens)
            for tok_id in word_token_ids:
                # 1) input ids and indices for mask token
                if np.random.rand()<self.mask_prob and (self.placeholder_token not in cap_word) and (cap_word not in self.prior_concept): 
                    masked_idxs.append(True)
                    input_ids_masked.append(self.mask_token_ids)
                else:
                    masked_idxs.append(False)
                    input_ids_masked.append(tok_id)

                # 2) keyword indices and labels for MLM
                if self.placeholder_token in cap_word:
                    is_keyword_tokens_mlm.append(True)
                    mlm_labels.append(-100)
                else:
                    is_keyword_tokens_mlm.append(False)
                    mlm_labels.append(tok_id)
        # 3) is_keyword_tokens_mlm - keyword indices for MLM
        for _ in range(len(is_keyword_tokens_mlm),self.tokenizer.model_max_length):
            is_keyword_tokens_mlm.append(False)
        assert len(is_keyword_tokens_mlm)==self.tokenizer.model_max_length
        example["is_keyword_tokens_mlm"]=torch.BoolTensor(is_keyword_tokens_mlm)
        # 4) input_ids or MLM
        input_ids_masked=input_ids_masked[:self.tokenizer.model_max_length-1]
        input_ids_masked.append(self.tokenizer.eos_token_id)
        for _ in range(len(input_ids_masked),self.tokenizer.model_max_length):
            input_ids_masked.append(self.tokenizer.pad_token_id)
        input_ids_masked=torch.LongTensor(input_ids_masked)
        example["input_ids_masked"]=input_ids_masked
        # 5) mlm_labels
        mlm_labels=mlm_labels[:self.tokenizer.model_max_length-1]
        if self.mlm_target not in ['non_special']: #if all learned (bos+eos+nonspecial) then add eos token
            mlm_labels.append(self.tokenizer.eos_token_id)
        for _ in range(len(mlm_labels),self.tokenizer.model_max_length):
            if self.mlm_target=='all':
                mlm_labels.append(self.tokenizer.pad_token_id)
            else: # non_padding/non_special/masked
                mlm_labels.append(-100)
        mlm_labels=torch.LongTensor(mlm_labels)
        example['mlm_labels']=mlm_labels

        
        # 6) non_special_idxs/masked_idxs
        masked_idxs=masked_idxs[:self.tokenizer.model_max_length-1]
        for _ in range(len(masked_idxs),self.tokenizer.model_max_length):
            masked_idxs.append(False)
        non_special_idxs=non_special_idxs[:self.tokenizer.model_max_length-1]
        for _ in range(len(non_special_idxs),self.tokenizer.model_max_length):
            non_special_idxs.append(False)
        masked_idxs=torch.BoolTensor(masked_idxs)
        non_special_idxs=torch.BoolTensor(non_special_idxs)
        example['masked_idxs']=masked_idxs
        example['non_special_idxs']=non_special_idxs
        return example

