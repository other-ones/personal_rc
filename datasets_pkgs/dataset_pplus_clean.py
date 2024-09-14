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
import spacy
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
prefixes=sorted(set([
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
]))

mlm_prefixes=sorted(set([
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of my {}",
    "a close-up photo of a {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a rendition of a {}",
]))

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

class PPlusDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        include_prior_concept,
        size=512,
        interpolation="bicubic",
        flip_p=0.5,
        center_crop=False,
        mlm_target='all',
        mask_prob=0.15,
        mask_token_ids=None,
        get_images=True,
        train_prior_concept1=None,
        placeholder_tokens=None,
        caption_root=None,
        seed=None,
        exclude_cap_types=None,
        prompt_type=None,
        target_image=None,
        check_tag=None,
    ):
        self.check_tag=check_tag
        self.nlp = spacy.load("en_core_web_sm")
        self.include_prior_concept=include_prior_concept
        if seed:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # if use multi-GPU
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print('seeded')

        self.exclude_cap_types=exclude_cap_types
        caption_dir_path=os.path.join(caption_root,prompt_type)
        cap_file_list=sorted(os.listdir(caption_dir_path))
        self.captions={}
        invalid_counts=0
        for cap_file in cap_file_list:
            fname=cap_file.split('.')[0]
            valid=True
            if exclude_cap_types is not None:
                for ect in exclude_cap_types:
                    if ect in fname:
                        valid=False
                        invalid_counts+=1
            if not valid:
                continue
            cap_file_path=os.path.join(caption_dir_path,cap_file)
            self.captions[fname]=sorted(list(set(open(cap_file_path).readlines()))) #CHECK
            print('{}\t{}'.format(fname,len(self.captions[fname])))
        if exclude_cap_types is not None:
            if len(exclude_cap_types)!=invalid_counts:
                print(invalid_counts,'invalid_counts',exclude_cap_types,'exclude_cap_types')
            assert len(exclude_cap_types)==invalid_counts,'len(exclude_cap_types)==invalid_counts'

        self._length=int(1e7)
        self.caption_types=sorted(list(self.captions.keys()))
        self.get_images = get_images

        if mask_token_ids is not None:
            mask_token_ids = mask_token_ids[0]
        self.mask_token_ids=mask_token_ids

        self.mask_prob = mask_prob
        self.mlm_target = mlm_target
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.train_prior_concept1 = train_prior_concept1
        self.placeholder_tokens = placeholder_tokens
        self.size = size
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.image_paths=[]
        for file_path in os.listdir(self.data_root):
            if (target_image is not None) and (target_image.split('.')[0] not in file_path):
                continue
            self.image_paths.append(os.path.join(self.data_root,file_path))
        self.image_paths=sorted(self.image_paths)


        self.num_images = len(self.image_paths)
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
            image = Image.open(img_path)
            if not image.mode == "RGB":
                image = image.convert("RGB")

            if np.random.rand()<0.5:
                image=image.transpose(Image.FLIP_LEFT_RIGHT)
           
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

            # 2. Caption for Pplus
            # 2.1 input_ids
            input_ids_list=[]
            for pidx in range(len(self.placeholder_tokens)):
                placeholder_string = self.placeholder_tokens[pidx]
                if self.include_prior_concept:
                    text = random.choice(prefixes).format(placeholder_string)+' {}'.format(self.train_prior_concept1)
                else:
                    text = random.choice(prefixes).format(placeholder_string)
                input_ids = self.tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids[0] #77
                input_ids_list.append(input_ids)
            input_ids_list=torch.stack(input_ids_list) # 9,77
            example["input_ids_list"]=input_ids_list
            example["raw_caption_ti"]=text

            # 2.1 input_ids
            # mlm data
            example["raw_caption_mlm"]=[]
            example["input_ids_pos"]=[]
            example["mlm_labels"]=[]
            example['masked_idxs']=[]
            example['non_special_idxs']=[]
            # mlm data

        else:




            # 3. MLM: 
            sampled_type=np.random.choice(self.caption_types)
            mlm_caption_raw=self.captions[sampled_type][index%len(self.captions[sampled_type])]
            mlm_caption_raw=mlm_caption_raw.strip()
            if 'interactions' not in sampled_type and 'creatives' not in sampled_type and 'specific' not in sampled_type:
                mlm_prefix=np.random.choice(mlm_prefixes)
                mlm_caption_raw=mlm_prefix.format(mlm_caption_raw)
            example["raw_caption_mlm"]=mlm_caption_raw
            example["input_ids_pos"] = self.tokenizer(
                    mlm_caption_raw,
                    padding="max_length",
                    truncation=True,
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids[0]
            # caption_mlm_raw,_=self.prompt_generator.generate_triplet()
            input_ids_masked_list=[]
            mlm_labels_list=[]
            masked_idxs_list=[]
            non_special_idxs_list=[]
            input_ids_non_mask_list=[]
            for pidx in range(len(self.placeholder_tokens)):
                mlm_caption=mlm_caption_raw.replace('<new1>','{}'.format(self.placeholder_tokens[pidx])) # caption without masked embedding
                words_mlm=mlm_caption.split()
                masked_idxs=[False] # DO NOT MASK SOT AND EOS token
                if self.mlm_target=='non_special': # if non-special only then bos is not learned
                    mlm_labels=[-100]
                else:
                    # masked,all
                    mlm_labels=[self.tokenizer.bos_token_id]
                input_ids_masked=[self.tokenizer.bos_token_id]
                non_special_idxs=[False]
                for word_idx in range(len(words_mlm)):
                    mlm_word=words_mlm[word_idx]
                    if self.check_tag is not None:
                        doc = self.nlp(mlm_word)
                        valid_mlm_tag=False
                        if mlm_word not in ['the','a','an']:
                            for token in doc:
                                pos_tag = token.pos_
                                if pos_tag in self.check_tag:
                                    valid_mlm_tag=True
                                    break
                    else:
                        valid_mlm_tag=True
                    word_token_ids=self.tokenizer.encode(mlm_word,add_special_tokens=False)
                    num_tokens=len(word_token_ids)
                    non_special_idxs+=([True]*num_tokens)
                    for tok_id in word_token_ids:
                        tok_decoded=self.tokenizer.decode(tok_id)
                        if valid_mlm_tag:
                            if np.random.rand()<self.mask_prob and (self.placeholder_token != mlm_word) and (mlm_word != self.train_prior_concept1): 
                                masked_idxs.append(True)
                                input_ids_masked.append(self.mask_token_ids)
                                mlm_labels.append(tok_id)
                            else:
                                masked_idxs.append(False)
                                input_ids_masked.append(tok_id)
                                if self.mlm_target in ['all','non_padding','non_special']:
                                    # all/non_padding/non_special
                                    mlm_labels.append(tok_id)
                                else:
                                    # masked
                                    mlm_labels.append(-100)
                        else:
                            # if non-target tag,
                            # do not contribute to mlm loss
                            masked_idxs.append(False)
                            input_ids_masked.append(tok_id)
                            mlm_labels.append(-100)
                            
                        # # 1) input ids and indices for mask token
                        # if np.random.rand()<self.mask_prob and (self.placeholder_tokens[pidx] != mlm_word) and (mlm_word != self.train_prior_concept1): 
                        #     masked_idxs.append(True)
                        #     input_ids_masked.append(self.mask_token_ids)
                        # else:
                        #     masked_idxs.append(False)
                        #     input_ids_masked.append(tok_id)
                        
                
                

                # 4) input_ids or MLM
                # Append EOS Token
                # In case 77th token is appended, remove it and replace with EOS token
                input_ids_masked=input_ids_masked[:self.tokenizer.model_max_length-1]
                input_ids_masked.append(self.tokenizer.eos_token_id) # FOR EOS
                for _ in range(len(input_ids_masked),self.tokenizer.model_max_length):
                    input_ids_masked.append(self.tokenizer.pad_token_id) # FOR PADDING
                input_ids_masked=torch.LongTensor(input_ids_masked)
                example["input_ids_masked"]=input_ids_masked


                # 5) mlm_labels
                # Append EOS Token
                # In case 77th token is appended, remove it and replace with EOS token
                mlm_labels=mlm_labels[:self.tokenizer.model_max_length-1]
                mlm_labels.append(-100) # FOR EOS
                for _ in range(len(mlm_labels),self.tokenizer.model_max_length):
                    mlm_labels.append(-100) # FOR PADDING


                # if self.mlm_target in ['all','non_padding']: 
                #     # We DO learn EOS 
                #     # IF MLM Target is all/non_padding
                #     mlm_labels.append(self.tokenizer.eos_token_id)
                # else:
                #     # masked/non_special
                #     # We do not learn EOS 
                #     # IF MLM Target is masked/non_special
                
                # for _ in range(len(mlm_labels),self.tokenizer.model_max_length):
                #     if self.mlm_target=='all':
                #         # Learn PADDING if MLM target is "all"
                #         mlm_labels.append(self.tokenizer.pad_token_id)
                #     else: 
                #         # non_padding/masked/non_special
                #         mlm_labels.append(-100)
                assert len(mlm_labels)==self.tokenizer.model_max_length
                mlm_labels=torch.LongTensor(mlm_labels)
                mlm_labels_list.append(mlm_labels)
                # 5) mlm_labels

                
                # 6) non_special_idxs/masked_idxs
                # Append EOS Token
                # In case last token is appended, remove it and replace with EOS token
                masked_idxs=masked_idxs[:self.tokenizer.model_max_length-1] 
                for _ in range(len(masked_idxs),self.tokenizer.model_max_length):
                    # We DO NOT mask EOS/PAD
                    masked_idxs.append(False)  # FOR EOS or PADDING
                masked_idxs=torch.BoolTensor(masked_idxs)
                masked_idxs_list.append(masked_idxs)

                # In case last token is appended, remove it and replace with EOS token
                non_special_idxs=non_special_idxs[:self.tokenizer.model_max_length-1]
                for _ in range(len(non_special_idxs),self.tokenizer.model_max_length):
                    non_special_idxs.append(False) #for EOS or PAD
                non_special_idxs=torch.BoolTensor(non_special_idxs)
                non_special_idxs_list.append(non_special_idxs)
                

                

            input_ids_masked_list=torch.stack(input_ids_masked_list)
            mlm_labels_list=torch.stack(mlm_labels_list)
            masked_idxs_list=torch.stack(masked_idxs_list)
            non_special_idxs_list=torch.stack(non_special_idxs_list)
            input_ids_non_mask_list=torch.stack(input_ids_non_mask_list)
            example['input_ids_masked_list']=input_ids_masked_list
            example['mlm_labels_list']=mlm_labels_list
            example['masked_idxs_list']=masked_idxs_list
            example['non_special_idxs_list']=non_special_idxs_list
            example['input_ids_non_mask_list']=input_ids_non_mask_list
        return example

