# Adapted from https://github.com/jmhessel/clipscore/blob/1036465276513621f77f1c2208d742e4a430781f/clipscore.py
'''
Code for CLIPScore (https://arxiv.org/abs/2104.08718)
@inproceedings{hessel2021clipscore,
  title={{CLIPScore:} A Reference-free Evaluation Metric for Image Captioning},
  author={Hessel, Jack and Holtzman, Ari and Forbes, Maxwell and Bras, Ronan Le and Choi, Yejin},
  booktitle={EMNLP},
  year={2021}
}
'''
import argparse
import clip
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
import tqdm
import numpy as np
import sklearn.preprocessing
import collections
import os
import pathlib
import json
import warnings
from packaging import version
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice


def get_all_metrics(refs, cands, return_per_cap=False):
    metrics = []
    names = []

    pycoco_eval_cap_scorers = [(Bleu(4), 'bleu'),
                               (Meteor(), 'meteor'),
                               (Rouge(), 'rouge'),
                               (Cider(), 'cider'),
                               (Spice(), 'spice')]

    for scorer, name in pycoco_eval_cap_scorers:
        overall, per_cap = pycoco_eval(scorer, refs, cands)
        if return_per_cap:
            metrics.append(per_cap)
        else:
            metrics.append(overall)
        names.append(name)

    metrics = dict(zip(names, metrics))
    return metrics


def tokenize(refs, cands, no_op=False):
    # no_op is a debug option to see how significantly not using the PTB tokenizer
    # affects things
    tokenizer = PTBTokenizer()

    if no_op:
        refs = {idx: [r for r in c_refs] for idx, c_refs in enumerate(refs)}
        cands = {idx: [c] for idx, c in enumerate(cands)}

    else:
        refs = {idx: [{'caption':r} for r in c_refs] for idx, c_refs in enumerate(refs)}
        cands = {idx: [{'caption':c}] for idx, c in enumerate(cands)}

        refs = tokenizer.tokenize(refs)
        cands = tokenizer.tokenize(cands)

    return refs, cands


def pycoco_eval(scorer, refs, cands):
    '''
    scorer is assumed to have a compute_score function.
    refs is a list of lists of strings
    cands is a list of predictions
    '''
    refs, cands = tokenize(refs, cands)
    average_score, scores = scorer.compute_score(refs, cands)
    return average_score, scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'candidates_json',
        type=str,
        help='Candidates json mapping from image_id --> candidate.')

    parser.add_argument(
        'image_dir',
        type=str,
        help='Directory of images, with the filenames as image ids.')

    parser.add_argument(
        '--references_json',
        default=None,
        help='Optional references json mapping from image_id --> [list of references]')

    parser.add_argument(
        '--compute_other_ref_metrics',
        default=1,
        type=int,
        help='If references is specified, should we compute standard reference-based metrics?')

    parser.add_argument(
        '--save_per_instance',
        default=None,
        help='if set, we will save per instance clipscores to this file')

    args = parser.parse_args()

    if isinstance(args.save_per_instance, str) and not args.save_per_instance.endswith('.json'):
        print('if you\'re saving per-instance, please make sure the filepath ends in json.')
        quit()
    return args


class CLIPCapDataset(torch.utils.data.Dataset):
    def __init__(self, data, prefix='A photo depicts'):
        self.data = data
        self.prefix = prefix
        if self.prefix[-1] != ' ':
            self.prefix += ' '

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(self.prefix + c_data, truncate=True).squeeze()
        return {'caption': c_data}

    def __len__(self):
        return len(self.data)


class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        if not os.path.exists(c_data):
            c_data=c_data.replace('.jpg','.png')
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {'image':image}

    def __len__(self):
        return len(self.data)


def extract_all_captions(captions, model, device, batch_size=256, num_workers=8):
    data = torch.utils.data.DataLoader(
        CLIPCapDataset(captions),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_text_features = []
    with torch.no_grad():
        # for b in tqdm.tqdm(data):
        for b in data:
            b = b['caption'].to(device)
            all_text_features.append(model.encode_text(b).cpu().numpy())
    all_text_features = np.vstack(all_text_features)
    return all_text_features


def extract_all_images(images, model, device, batch_size=64, num_workers=8):
    data = torch.utils.data.DataLoader(
        CLIPImageDataset(images),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_image_features = []
    with torch.no_grad():
        # for b in tqdm.tqdm(data):
        for b in data:
            b = b['image'].to(device)
            # print(b.shape,'b.shape')
            if device == 'cuda':
                b = b.to(torch.float16)
            all_image_features.append(model.encode_image(b).cpu().numpy())
            # print(model.encode_image(b).cpu().numpy().shape,'model.encode_image(b).cpu().numpy()shape')
    # print(len(all_image_features),'len(all_image_features)',type(all_image_features))
    # print(all_image_features[0].shape,'all_image_features[0].shape')
    # print(all_image_features[1].shape,'all_image_features[1].shape')
    all_image_features = np.vstack(all_image_features)
    # print(all_image_features.shape,'all_image_features.shape')
    return all_image_features


def get_clip_score(model, images, candidates, device, w=2.5):
    '''
    get standard image-text clipscore.
    images can either be:
    - a list of strings specifying filepaths for images
    - a precomputed, ordered matrix of image features
    '''
    if isinstance(images, list): # NOT HERE
        # need to extract image features
        images = extract_all_images(images, model, device)

    candidates = extract_all_captions(candidates, model, device)
    #as of numpy 1.21, normalize doesn't work properly for float16
    if version.parse(np.__version__) < version.parse('1.21'):
        images = sklearn.preprocessing.normalize(images, axis=1)
        candidates = sklearn.preprocessing.normalize(candidates, axis=1)
    else: #HERE
        warnings.warn(
            'due to a numerical instability, new numpy normalization is slightly different than paper results. '
            'to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')
        # (100, 512) images.shape
        # print(np.linalg.norm(images[0]),'before')
        images = images / np.sqrt(np.sum(images**2, axis=1, keepdims=True))
        # print(np.linalg.norm(images[0]),'after')
        candidates = candidates / np.sqrt(np.sum(candidates**2, axis=1, keepdims=True))
    per = w*np.clip(np.sum(images * candidates, axis=1), 0, None)
    # print(np.sum(images[0]),'images[0]')
    # print(np.sum(candidates[0]),'candidates[0]')
    # print(np.mean(per),'per')
    # print(w,'weight')
    return np.mean(per), per, candidates




def cal_clipscore(image_ids, image_paths, text_list, device=None, scale_weight=1):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()
    image_paths=sorted(image_paths)
    image_feats = extract_all_images(image_paths, model, device, batch_size=32, num_workers=8)
    # print(image_paths[0],'image_paths[0]')
    # print(np.sum(image_feats[0]),'image_feats_sum_td')
    # print(image_feats.shape,'image_feats.shape')
    # get image-text clipscore
    _, per_instance_image_text, candidate_feats = get_clip_score(model, image_feats, text_list, device, w=scale_weight)

    
    scores = {image_id: {'CLIPScore': float(clipscore)}
                for image_id, clipscore in
                zip(image_ids, per_instance_image_text)}
    # print('CLIPScore: {:.4f}'.format(np.mean([s['CLIPScore'] for s in scores.values()])))

    return scores

def main():
    args = parse_args()

    image_paths = [os.path.join(args.image_dir, path) for path in os.listdir(args.image_dir)
                   if path.endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
    image_ids = [pathlib.Path(path).stem for path in image_paths]

    with open(args.candidates_json) as f:
        candidates = json.load(f)
    candidates = [candidates[cid] for cid in image_ids]


    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == 'cpu':
        warnings.warn(
            'CLIP runs in full float32 on CPU. Results in paper were computed on GPU, which uses float16. '
            'If you\'re reporting results on CPU, please note this when you report.')
    model, transform = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()

    image_feats = extract_all_images(
        image_paths, model, device, batch_size=64, num_workers=8)

    # get image-text clipscore
    _, per_instance_image_text, candidate_feats = get_clip_score(
        model, image_feats, candidates, device)

    
    scores = {image_id: {'CLIPScore': float(clipscore)}
                for image_id, clipscore in
                zip(image_ids, per_instance_image_text)}
    print('CLIPScore: {:.4f}'.format(np.mean([s['CLIPScore'] for s in scores.values()])))

    

    if args.save_per_instance:
        with open(args.save_per_instance, 'w') as f:
            f.write(json.dumps(scores))


if __name__ == '__main__':
    main()
