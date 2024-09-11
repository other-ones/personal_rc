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

junctions=[
            "displayed alongside",
            "displayed next to",
            "exhibited with",
            "exhibited next to",
            "shown next to",
            "presented with",
            "featured with",
            "positioned beside",
            "accompanied with",
            "placed adjacent to",
            "arranged with",
            "paired alongside",
            "set against",
            "aligned with",
            "juxtaposed with",
            "matched with",
            "grouped with",
            "associated next to",
            "framed with"
            "with",
            "next to",
            "shown with",
            "alongside",
            "shown next to",
            "adjacent to",
            "together with",
            "accompanied by",
            "in proximity to",
            "in collaboration with",
            "associated with",
            "in partnership with",
            "in relation to",
            "displayed with",
            "near by"
                        ]
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

class TextualInversionDatasetMulti(Dataset):
    def __init__(
        self,
        
        tokenizer,
        include_prior_concept,
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0,
        center_crop=False,
        exclude_suffix=True,
        mlm_target='all',
        mask_prob=0.15,
        mask_token_ids=None,
        get_images=True,
        prompt_type=None,
        
        
        # multi
        placeholder_tokens=[],
        placeholder_ids=[],
        data_root1=None,
        data_root2=None,
        prior_concepts="*",
        make_composition=True,

        # prior_preservation
        class_num=None,
        class_data_root1=None,
        class_data_root2=None,
        class_prompt1=None,
        class_prompt2=None,
        simple_caption=False,
        mlm_prior=0,

    ):
        # multi
        self.placeholder_tokens = placeholder_tokens
        self.placeholder_ids = placeholder_ids
        self.data_root1 = data_root1
        self.data_root2 = data_root2
        self.prior_concepts = prior_concepts
        self.make_composition = make_composition
        print(placeholder_tokens,'placeholder_tokens')
        print(prior_concepts,'prior_concepts')

        # prior_preservation
        self.class_num = class_num
        self.class_data_root1 = class_data_root1
        self.class_data_root2 = class_data_root2
        self.class_prompt1 = class_prompt1
        self.class_prompt2 = class_prompt2
        self.simple_caption = simple_caption
        self.mlm_prior = mlm_prior
        self.instance_images_path = list(Path(data_root1).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        
        if class_data_root1 is not None:
            self.class_data_root1 = Path(class_data_root1)
            self.class_data_root2 = Path(class_data_root2)
            self.class_data_root1.mkdir(parents=True, exist_ok=True)
            self.class_data_root2.mkdir(parents=True, exist_ok=True)
            self.class_images_path1 = list(self.class_data_root1.iterdir())
            self.class_images_path2 = list(self.class_data_root2.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path1), class_num)
            else:
                self.num_class_images = len(self.class_images_path1)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt1 = class_prompt1
            self.class_prompt2 = class_prompt2
            self.class_prompts=[self.class_prompt1,self.class_prompt2]
            self.class_images_paths=[self.class_images_path1,self.class_images_path2]
        else:
            self.class_prompt1 = None
            self.class_prompt2 = None
            self.class_data_root1 = None
            self.class_data_root2 = None
            self.class_images_path1 = None
            self.class_images_path2 = None
            self.class_prompts=None
            self.class_images_paths=None
        print(class_data_root1,'class_data_root1')
        # prior_preservation
        


        self.prompt_type=prompt_type
        self.include_prior_concept=include_prior_concept
        if prompt_type=='two_pets':
            from .mlm_pkgs.caption_generator_two_pets import CaptionGeneratorTwoPets
            self.prompt_generator_multi=CaptionGeneratorTwoPets()
            from .mlm_pkgs.caption_generator_pet import CaptionGeneratorPet
            self.prompt_generator_single=CaptionGeneratorPet()
        else:
            raise Exception('Not Implemented')
        
        self.get_images = get_images
        self.mask_token_ids = mask_token_ids
        self.mask_prob = mask_prob
        self.mlm_target = mlm_target
        self.exclude_suffix = exclude_suffix
        self.tokenizer = tokenizer
        # self.prior_concept_id=tokenizer.encode(self.prior_concept,add_special_tokens=False)[0]
        self.size = size
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.image_paths1 = [os.path.join(self.data_root1, file_path) for file_path in os.listdir(self.data_root1)]
        self.image_paths2 = [os.path.join(self.data_root2, file_path) for file_path in os.listdir(self.data_root2)]
        self.image_paths=[self.image_paths1,self.image_paths2]
        self.num_images = len(self.image_paths1)+len(self.image_paths2)

        self._length = self.num_images * repeats
        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        # self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)
        self.junctions=list(set(junctions))
    def __len__(self):
        return self._length

    def compose_images(self,images,masks):
        dst_img=np.zeros((512,512,3))
        dst_mask=np.zeros((512,512))
        idxs=[0,1]
        np.random.shuffle(idxs)
        img1=images[idxs[0]]
        mask1=masks[idxs[0]]
        img2=images[idxs[1]]
        mask2=masks[idxs[1]]
        if np.random.rand()<0.5:
            # left-right split
            top1=np.random.randint(0,256)
            top2=np.random.randint(0,256)
            dst_img[top1:top1+256,:256,:]=img1
            dst_img[top2:top2+256,256:,:]=img2
            dst_mask[top1:top1+256,:256]=mask1
            dst_mask[top2:top2+256,256:]=mask2
        else:
            # top-bottom split
            left1=np.random.randint(0,256)
            left2=np.random.randint(0,256)
            dst_img[:256,left1:left1+256,:]=img1
            dst_img[256:,left2:left2+256,:]=img2
            dst_mask[:256,left1:left1+256]=mask1
            dst_mask[256:,left2:left2+256]=mask2
        return dst_img,dst_mask
    def __getitem__(self, index):
        example = {}
        # 1. Image
        if self.get_images:
            # multi concept image
            if np.random.rand()<0.5 and self.make_composition:
                # sampled_concept=np.random.choice([0,1])
                img_path1=self.image_paths[0][index % (len(self.image_paths[0]))]
                img_path2=self.image_paths[1][index % (len(self.image_paths[1]))]
                mask_path1=img_path1.replace('/images/','/masks/')
                mask_path1=mask_path1.replace('.jpeg','.png')
                mask_path1=mask_path1.replace('.jpg','.png')
                mask_path2=img_path2.replace('/images/','/masks/')
                mask_path2=mask_path2.replace('.jpeg','.png')
                mask_path2=mask_path2.replace('.jpg','.png')
                image1 = cv2.imread(img_path1)
                image1=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
                image1=cv2.resize(image1,(256,256))
                image2 = cv2.imread(img_path2)
                image2=cv2.cvtColor(image2,cv2.COLOR_BGR2RGB)
                image2=cv2.resize(image2,(256,256))
                
                mask1=cv2.imread(mask_path1,-1)
                mask2=cv2.imread(mask_path2,-1)
                mask1=cv2.resize(mask1,(256,256),cv2.INTER_NEAREST)
                mask2=cv2.resize(mask2,(256,256),cv2.INTER_NEAREST)
                if np.random.rand()<0.5:
                    image1=cv2.flip(image1,1)
                    mask1=cv2.flip(mask1,1)
                if np.random.rand()<0.5:
                    image2=cv2.flip(image2,1)
                    mask2=cv2.flip(mask2,1)
                image,mask=self.compose_images([image1,image2],[mask1,mask2])
                # 2. Caption for TI
                prior_concept1 = self.prior_concepts[0]
                prior_concept2 = self.prior_concepts[1]
                placeholder_token1=self.placeholder_tokens[0]
                placeholder_token2=self.placeholder_tokens[1]
                
                if self.include_prior_concept:
                    placeholder1='{} {}'.format(placeholder_token1,prior_concept1)
                    placeholder2='{} {}'.format(placeholder_token2,prior_concept2)
                else:
                    placeholder1='{}'.format(placeholder_token1)
                    placeholder2='{}'.format(placeholder_token2)
                sampled_junction=np.random.choice(self.junctions)
                if np.random.rand()<0.5:
                    text = random.choice(prefixes).format(placeholder1+' {} '.format(sampled_junction)+placeholder2)
                else:
                    text = random.choice(prefixes).format(placeholder2+' {} '.format(sampled_junction)+placeholder1)
            else:
                # single concept image
                sampled_concept=np.random.choice([0,1])
                img_path=self.image_paths[sampled_concept][index % (len(self.image_paths[sampled_concept]))]
                image = cv2.imread(img_path)
                image=cv2.resize(image,(512,512))
                image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                mask_path=img_path.replace('/images/','/masks/')
                mask_path=mask_path.replace('.jpeg','.png')
                mask_path=mask_path.replace('.jpg','.png')
                mask=cv2.imread(mask_path,-1)
                mask=cv2.resize(mask,(512,512),cv2.INTER_NEAREST)
                if np.random.rand()<0.5:
                    image=cv2.flip(image,1)
                    mask=cv2.flip(mask,1)
                # 2. Caption for TI
                prior_concept = self.prior_concepts[sampled_concept]
                placeholder_token=self.placeholder_tokens[sampled_concept]
                if self.include_prior_concept:
                    placeholder='{} {}'.format(placeholder_token,prior_concept)
                else:
                    placeholder='{}'.format(placeholder_token)
                text = random.choice(prefixes).format(placeholder)
            # prior_image
            if self.class_images_paths:
                sampled_prior=np.random.choice([0,1])
                num_class_images=len(self.class_images_paths[sampled_prior])
                class_image = Image.open(self.class_images_paths[sampled_prior][index % num_class_images])
                if not class_image.mode == "RGB":
                    class_image = class_image.convert("RGB")
                # class_image = exif_transpose(class_image)
                class_image = class_image.resize((self.size, self.size), resample=self.interpolation)
                class_image = np.array(class_image).astype(np.uint8)
                class_image = (class_image / 127.5 - 1.0).astype(np.float32)
                class_image=torch.from_numpy(class_image).permute(2, 0, 1)
                example["class_images"] = class_image
                class_text_inputs=self.tokenizer(
                    self.class_prompts[sampled_prior],
                    truncation=True,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                )
                example["class_prompt_ids"] = class_text_inputs.input_ids[0]
                is_keyword_tokens_prior=[False]*self.tokenizer.model_max_length
                assert len(is_keyword_tokens_prior)==self.tokenizer.model_max_length
                is_keyword_tokens_prior=torch.BoolTensor(is_keyword_tokens_prior)
                example["is_keyword_tokens_prior"]=is_keyword_tokens_prior
            # prior_image
                
                
            img = image.astype(np.uint8)
            if self.center_crop: #NO
                crop = min(img.shape[0], img.shape[1])
                (h,w,) = (img.shape[0],img.shape[1],)
                img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]
            image = Image.fromarray(img)
            image = np.array(image).astype(np.uint8)
            image = (image / 127.5 - 1.0).astype(np.float32)
            mask_tensor=torch.from_numpy(mask).unsqueeze(-1).permute(2, 0, 1) # binary mask
            example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
            example["masks"] = mask_tensor
            example["input_ids"] = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]
            is_keyword_tokens1=[False]
            is_keyword_tokens2=[False]
            text_words=text.split()
            for word_idx in range(len(text_words)):
                cap_word=text_words[word_idx]
                word_token_ids=self.tokenizer.encode(cap_word,add_special_tokens=False)
                num_tokens=len(word_token_ids)
                for tok_id in word_token_ids:
                    if cap_word in self.placeholder_tokens[0]:
                        is_keyword_tokens1.append(True)
                    else:
                        is_keyword_tokens1.append(False)
                    if cap_word in self.placeholder_tokens[1]:
                        is_keyword_tokens2.append(True)
                    else:
                        is_keyword_tokens2.append(False)

            # 3) is_keyword_tokens - keyword indices for MLM
            for _ in range(len(is_keyword_tokens1),self.tokenizer.model_max_length):
                is_keyword_tokens1.append(False)
            assert len(is_keyword_tokens1)==self.tokenizer.model_max_length
            for _ in range(len(is_keyword_tokens2),self.tokenizer.model_max_length):
                is_keyword_tokens2.append(False)
            assert len(is_keyword_tokens2)==self.tokenizer.model_max_length
            example["is_keyword_tokens1"]=torch.BoolTensor(is_keyword_tokens1)
            example["is_keyword_tokens2"]=torch.BoolTensor(is_keyword_tokens2)
        # 3. MLM
        if np.random.rand()<0.5:
            caption_mlm=self.prompt_generator_multi.generate_caption()
        else:
            caption_mlm=self.prompt_generator_single.generate_caption()
        sampled_junction_mlm=np.random.choice(self.junctions)
        caption_mlm=caption_mlm.replace('[JUNCTION]',sampled_junction_mlm)
        


        if self.include_prior_concept:
            placeholder1='{} {}'.format(self.placeholder_tokens[0],self.prior_concepts[0])
            placeholder2='{} {}'.format(self.placeholder_tokens[1],self.prior_concepts[1])
        else:
            placeholder1='{}'.format(self.placeholder_tokens[0])
            placeholder2='{}'.format(self.placeholder_tokens[1])
        if np.random.rand()<0.5:
            caption_mlm=caption_mlm.format(placeholder1,placeholder2)
        else:
            caption_mlm=caption_mlm.format(placeholder2,placeholder1)
        example["input_ids_pos"] = self.tokenizer(
                caption_mlm,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]
        

        
        caption_single1=self.prompt_generator_single.generate_caption()
        caption_single2=self.prompt_generator_single.generate_caption()
        caption_single1=caption_single1.format(placeholder1)
        caption_single2=caption_single2.format(placeholder2)
        example["input_ids_single1"] = self.tokenizer(
                caption_single1,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]
        example["input_ids_single2"] = self.tokenizer(
                caption_single2,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]
        
        is_keyword_tokens_single1=torch.zeros(self.tokenizer.model_max_length)
        is_keyword_tokens_single1[example["input_ids_single1"]==self.placeholder_ids[0]]=1
        is_keyword_tokens_single1=is_keyword_tokens_single1.bool()
        example["is_keyword_tokens_single1"]=is_keyword_tokens_single1

        is_keyword_tokens_single2=torch.zeros(self.tokenizer.model_max_length)
        is_keyword_tokens_single2[example["input_ids_single2"]==self.placeholder_ids[1]]=1
        is_keyword_tokens_single2=is_keyword_tokens_single2.bool()
        example["is_keyword_tokens_single2"]=is_keyword_tokens_single2
        
        
        words_mlm=caption_mlm.split()
        is_keyword_tokens_mlm1=[False] # first token for <startoftext>
        is_keyword_tokens_mlm2=[False] # first token for <startoftext>
        masked_idxs=[False]
        if self.mlm_target=='non_special': # if non-special only then bos is not learned
            mlm_labels=[-100]
        else:
            mlm_labels=[self.tokenizer.bos_token_id]
        input_ids_masked=[self.tokenizer.bos_token_id]
        non_special_idxs=[False]
        for word_idx in range(len(words_mlm)):
            cap_word=words_mlm[word_idx]
            word_token_ids=self.tokenizer.encode(cap_word,add_special_tokens=False)
            num_tokens=len(word_token_ids)
            non_special_idxs+=([True]*num_tokens)
            for tok_id in word_token_ids:
                # 1) input ids and indices for mask token
                if np.random.rand()<self.mask_prob and (cap_word not in self.placeholder_tokens) and (cap_word not in self.prior_concepts): 
                    masked_idxs.append(True)
                    input_ids_masked.append(self.mask_token_ids)
                else:
                    masked_idxs.append(False)
                    input_ids_masked.append(tok_id)
                # 2) keyword indices and labels for MLM
                if cap_word in self.placeholder_tokens:
                    assert num_tokens==1
                    if cap_word == self.placeholder_tokens[0]:
                        is_keyword_tokens_mlm1.append(True)
                    else:
                        is_keyword_tokens_mlm1.append(False)
                    if cap_word == self.placeholder_tokens[1]:
                        is_keyword_tokens_mlm2.append(True)
                    else:
                        is_keyword_tokens_mlm2.append(False)
                    mlm_labels.append(-100)
                else:
                    mlm_labels.append(tok_id)
                    is_keyword_tokens_mlm1.append(False)
                    is_keyword_tokens_mlm2.append(False)
                
                
                    
         

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

        
        # 7) is_keyword_tokens
        for _ in range(len(is_keyword_tokens_mlm1),self.tokenizer.model_max_length):
            is_keyword_tokens_mlm1.append(False)
        for _ in range(len(is_keyword_tokens_mlm2),self.tokenizer.model_max_length):
            is_keyword_tokens_mlm2.append(False)
        example["is_keyword_tokens_mlm1"]=torch.BoolTensor(is_keyword_tokens_mlm1)
        example["is_keyword_tokens_mlm2"]=torch.BoolTensor(is_keyword_tokens_mlm2)
        return example

