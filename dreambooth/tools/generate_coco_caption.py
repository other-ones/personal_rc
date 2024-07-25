import os
import numpy as np
import json

caption_json_path='/data/dataset/coco/karpathy/dataset_coco.json'





catid2name={1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
catid2supcat={1: 'person', 2: 'vehicle', 3: 'vehicle', 4: 'vehicle', 5: 'vehicle', 6: 'vehicle', 7: 'vehicle', 8: 'vehicle', 9: 'vehicle', 10: 'outdoor', 11: 'outdoor', 13: 'outdoor', 14: 'outdoor', 15: 'outdoor', 16: 'animal', 17: 'animal', 18: 'animal', 19: 'animal', 20: 'animal', 21: 'animal', 22: 'animal', 23: 'animal', 24: 'animal', 25: 'animal', 27: 'accessory', 28: 'accessory', 31: 'accessory', 32: 'accessory', 33: 'accessory', 34: 'sports', 35: 'sports', 36: 'sports', 37: 'sports', 38: 'sports', 39: 'sports', 40: 'sports', 41: 'sports', 42: 'sports', 43: 'sports', 44: 'kitchen', 46: 'kitchen', 47: 'kitchen', 48: 'kitchen', 49: 'kitchen', 50: 'kitchen', 51: 'kitchen', 52: 'food', 53: 'food', 54: 'food', 55: 'food', 56: 'food', 57: 'food', 58: 'food', 59: 'food', 60: 'food', 61: 'food', 62: 'furniture', 63: 'furniture', 64: 'furniture', 65: 'furniture', 67: 'furniture', 70: 'furniture', 72: 'electronic', 73: 'electronic', 74: 'electronic', 75: 'electronic', 76: 'electronic', 77: 'electronic', 78: 'appliance', 79: 'appliance', 80: 'appliance', 81: 'appliance', 82: 'appliance', 84: 'indoor', 85: 'indoor', 86: 'indoor', 87: 'indoor', 88: 'indoor', 89: 'indoor', 90: 'indoor'}



print(list(set(list(catid2supcat.values()))))
image_id_to_catlist={}
image_id_to_supcatlist={}
json_count=0
for mode in ['train','val']:
    det_json_path='/data/dataset/coco/detection/annotations/instances_{}2014.json'.format(mode)
    print('loading det',det_json_path)
    det_data=json.load(open(det_json_path))
    det_ann_list=det_data['annotations']
    cat_info=det_data['categories']
    mode_ids=[]
    for det_ann in det_ann_list:
        image_id=det_ann['image_id']
        catid=det_ann['category_id']
        mode_ids.append(image_id)
        supcat=catid2supcat[catid]
        if image_id in image_id_to_catlist:
            if catid not in image_id_to_catlist[image_id]:
                image_id_to_catlist[image_id].append(catid)
        else:
            image_id_to_catlist[image_id]=[catid]
        if image_id in image_id_to_supcatlist:
            if supcat not in image_id_to_supcatlist[image_id]:
                image_id_to_supcatlist[image_id].append(supcat)
        else:
            image_id_to_supcatlist[image_id]=[supcat]
    json_count+=len(set(mode_ids))

dst_root='/data/dataset/coco'

print(np.min(list(image_id_to_supcatlist.keys())),np.max(list(image_id_to_supcatlist.keys())),'minmax')
caption_data=json.load(open(caption_json_path))       
cat_ann_list=caption_data['images']
count=0
all_in_count=0
for cat_ann in cat_ann_list:
    image_id=cat_ann['cocoid']
    filename=cat_ann['filename']
    sentences=cat_ann['sentences']
    raw_captions=[item['raw'] for item in sentences]
    if not image_id in image_id_to_catlist:
        continue
    catlist=image_id_to_catlist[image_id]
    supcatlist=image_id_to_supcatlist[image_id]
    catnames=[catid2name[item]for item in catlist]
    
    # catnames=set(catnames)
    for raw_cap in raw_captions:
        raw_cap=raw_cap.lower()
        print(raw_cap,catnames,supcatlist)
    #     if raw_cap.endswith('.'):
    #         raw_cap=raw_cap[:-1]
    #     cap_words=raw_cap.split()
    #     for catname in catnames:
    #         for word in cap_words:
    #             if word==catname:
    #                 count+=1
    #                 # print(word,'word','catname',raw_cap)
    # if not all_in:
    #     print(raw_cap,'raw_cap')
    #     print(catnames,'catnames')



    
    

print(count,'count',len(image_id_to_supcatlist),'image_id_to_supcatlist')
print(count,'count',len(image_id_to_catlist),'image_id_to_catlist')
print(count,'count',len(cat_ann_list),'cat_ann_list')
print(json_count,'json_count')
print(all_in_count,'all_in_count')

