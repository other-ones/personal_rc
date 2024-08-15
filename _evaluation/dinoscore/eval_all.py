import json
import os
from pathlib import Path
import torchmetrics
from collections import defaultdict
from accelerate import Accelerator
from transformers import ViTImageProcessor, ViTModel
import torch
from torch.nn.functional import cosine_similarity
from PIL import Image

class DINOEvaluator:
    def __init__(self, device, dino_model='facebook/dino-vits16') -> None:
        self.device = device
        self.processor = ViTImageProcessor.from_pretrained(dino_model)
        self.model = ViTModel.from_pretrained(dino_model).to(device)

    @torch.inference_mode()
    def get_image_features(self, images) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(device=self.device)
        features = self.model(**inputs).last_hidden_state[:, 0, :]
        return features

    @torch.inference_mode()
    def img_to_img_similarity(self, src_images, generated_images):
        src_features = self.get_image_features(src_images)
        gen_features = self.get_image_features(generated_images)

        return torchmetrics.functional.pairwise_cosine_similarity(src_features, gen_features).mean().item()

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--method_root',required=True)
    args=parser.parse_args()
    method_root=args.method_root
    result_root=os.path.join(method_root,'results')
    dirs=os.listdir(result_root)
    dino_eval=DINOEvaluator(device='cuda:0')
    dirs=sorted(dirs)
    for dir in dirs:
        dir_path=os.path.join(result_root,dir)
        concepts=os.listdir(dir_path)
        for concept in concepts:
            concept_path=os.path.join(dir_path,concept)
            exps=os.listdir(concept_path)
            real_root=os.path.join('/data/twkim/diffusion/personalization/collected/images/',concept)
            for exp in exps:
                exp_path=os.path.join(concept_path,exp)
                log_path=os.path.join(exp_path,'result.txt')
                fake_root=os.path.join(exp_path,'generated')
                dst_path=os.path.join(exp_path,'dino.json')
                if os.path.exists(dst_path):
                    read_data=json.load(open(dst_path))
                    print('{}\t{}'.format(exp_path,read_data['dino']))
                    continue
                if not (os.path.exists(log_path)):
                    continue
                if not os.path.exists(fake_root):
                    continue
                src_images=[Image.open(os.path.join(real_root,item)).convert('RGB').resize((512,512)) for item in os.listdir(real_root)]
                generated_images=[]
                for item in os.listdir(fake_root):
                    if not item.endswith(('.png','.jpg')):
                        continue
                    generated_images.append(Image.open(os.path.join(fake_root,item)))
                if not len(generated_images):
                    continue
                score=dino_eval.img_to_img_similarity(src_images=src_images,generated_images=generated_images)
                print('{}\t{}'.format(exp_path,score))
                dst_file=open(os.path.join(exp_path,'dino.json'),'w')
                dst_data={'dino':float(score)}
                json.dump(dst_data,dst_file)