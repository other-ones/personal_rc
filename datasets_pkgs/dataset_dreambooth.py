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

class DreamboothDataset(Dataset):
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
        class_num=None,
        class_data_root=None,
        class_prompt=None,
        simple_caption=False,

    ):
        self.simple_caption=simple_caption
        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_prompt = None
            self.class_data_root = None
            self.class_images_path = None


        self.prompt_type=prompt_type
        self.include_prior_concept=include_prior_concept
        caption_dir_path=os.path.join(caption_root,prompt_type)
        cap_file_list=os.listdir(caption_dir_path)
        self.captions={}
        max_length=0
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
        # 1. Image
        if self.get_images:
            img_path=self.image_paths[index % (len(self.image_paths))]
            mask_path=img_path.replace('/images/','/masks/')
            mask_path=mask_path.replace('.jpg','.png')
            mask_path=mask_path.replace('.jpeg','.png')
            mask=cv2.imread(mask_path,-1)
            mask=cv2.resize(mask,(512,512),cv2.INTER_NEAREST)
            image = Image.open(img_path)
            if not image.mode == "RGB":
                image = image.convert("RGB")

            if np.random.rand()<0.5:
                image=image.transpose(Image.FLIP_LEFT_RIGHT)
                mask=cv2.flip(mask,1)
           
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
            # image = self.flip_transform(image)
            image = np.array(image).astype(np.uint8)
            image = (image / 127.5 - 1.0).astype(np.float32)
            example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
            mask_tensor=torch.from_numpy(mask).unsqueeze(-1).permute(2, 0, 1) # binary mask
            example["masks"] = mask_tensor

            # prior_image
            if self.class_images_path:
                class_image = Image.open(self.class_images_path[index % self.num_class_images])
                if not class_image.mode == "RGB":
                    class_image = class_image.convert("RGB")
                # class_image = exif_transpose(class_image)
                class_image = class_image.resize((self.size, self.size), resample=self.interpolation)
                class_image = np.array(class_image).astype(np.uint8)
                class_image = (class_image / 127.5 - 1.0).astype(np.float32)
                class_image=torch.from_numpy(class_image).permute(2, 0, 1)
                example["class_images"] = class_image
                class_text_inputs=self.tokenizer(
                    self.class_prompt,
                    truncation=True,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                )
                example["class_prompt_ids"] = class_text_inputs.input_ids[0]
                is_keyword_tokens_prior=[False]
                # print(self.class_prompt,'self.class_prompt')
                text_words_prior=self.class_prompt.split()
                for word_idx in range(len(text_words_prior)):
                    cap_word=text_words_prior[word_idx]
                    word_token_ids=self.tokenizer.encode(cap_word,add_special_tokens=False)
                    num_tokens=len(word_token_ids)
                    for tok_id in word_token_ids:
                        if self.placeholder_token == cap_word:
                            is_keyword_tokens_prior.append(True)
                        else:
                            is_keyword_tokens_prior.append(False)
                for _ in range(len(is_keyword_tokens_prior),self.tokenizer.model_max_length):
                    is_keyword_tokens_prior.append(False)
                
                assert len(is_keyword_tokens_prior)==self.tokenizer.model_max_length
                is_keyword_tokens_prior=torch.BoolTensor(is_keyword_tokens_prior)
                example["is_keyword_tokens_prior"]=is_keyword_tokens_prior
            # prior_image


            # 2. Caption for TI
            placeholder_string = self.placeholder_token
            if self.simple_caption:
                prefix=prefixes[0]
            else:
                prefix=random.choice(prefixes)
            if self.include_prior_concept:
                text = prefix.format(placeholder_string)+' {}'.format(self.prior_concept)
            else:
                text = prefix.format(placeholder_string)
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
            assert sum(is_keyword_tokens)==1
            example["is_keyword_tokens"]=torch.BoolTensor(is_keyword_tokens)
        # 3. MLM
        # mlm_pos,mlm_neg=self.prompt_generator.generate_triplet()
        sampled_type=np.random.choice(self.caption_types)
        mlm_caption=self.captions[sampled_type][index%len(self.captions[sampled_type])]
        if self.include_prior_concept:
            placeholder='{} {}'.format(self.placeholder_token,self.prior_concept)
        else:
            placeholder='{}'.format(self.placeholder_token)
        mlm_caption=mlm_caption.replace('<new1>','{}'.format(placeholder)) # caption without masked embedding
        example["input_ids_pos"] = self.tokenizer(
                mlm_caption,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]
        
        


        words_mlm=mlm_caption.split()
        is_keyword_tokens_mlm=[False] # first token for <startoftext>
        masked_idxs=[False]
        if self.mlm_target=='non_special': # if non-special only then bos is not learned
            mlm_labels=[-100]
        else:
            mlm_labels=[self.tokenizer.bos_token_id]
        input_ids_masked=[self.tokenizer.bos_token_id]
        non_special_idxs=[False]
        for word_idx in range(len(words_mlm)):
            mlm_word=words_mlm[word_idx]
            word_token_ids=self.tokenizer.encode(mlm_word,add_special_tokens=False)
            num_tokens=len(word_token_ids)
            non_special_idxs+=([True]*num_tokens)
            for tok_id in word_token_ids:
                # 1) input ids and indices for mask token
                if np.random.rand()<self.mask_prob and (self.placeholder_token != mlm_word) and (mlm_word != self.prior_concept): 
                    masked_idxs.append(True)
                    input_ids_masked.append(self.mask_token_ids)
                else:
                    masked_idxs.append(False)
                    input_ids_masked.append(tok_id)
                # 2) keyword indices and labels for MLM
                if self.placeholder_token == mlm_word:
                    assert num_tokens==1
                    is_keyword_tokens_mlm.append(True)
                    mlm_labels.append(-100)
                else:
                    is_keyword_tokens_mlm.append(False)
                    mlm_labels.append(tok_id)
         

        # 3) is_keyword_tokens_mlm - keyword indices for MLM
        for _ in range(len(is_keyword_tokens_mlm),self.tokenizer.model_max_length):
            is_keyword_tokens_mlm.append(False)
        assert len(is_keyword_tokens_mlm)==self.tokenizer.model_max_length
        assert sum(is_keyword_tokens_mlm)==1
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

