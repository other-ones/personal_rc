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
    parser.add_argument('--fake_root')
    parser.add_argument('--real_root')
    args=parser.parse_args()
    fake_root=args.fake_root
    real_root=args.real_root
    src_images=[Image.open(os.path.join(real_root,item)).convert('RGB').resize((512,512)) for item in os.listdir(real_root)]
    print(os.listdir(real_root),src_images[0].size)
    generated_images=[]
    for item in os.listdir(fake_root):
        if not item.endswith(('.png','.jpg')):
            continue
        generated_images.append(Image.open(os.path.join(fake_root,item)))
    dino_eval=DINOEvaluator(device='cuda:0')
    score=dino_eval.img_to_img_similarity(src_images=src_images,generated_images=generated_images)
    print(score)