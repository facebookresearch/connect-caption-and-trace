from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
from captioning.utils import misc as utils
# from .local_optimal_transport import local_OT

# load coco-caption if available
try:
    sys.path.append("coco-caption")
    sys.path.append("/home/zihang/Research/Localized_Narratives/ImageCaptioning.pytorch")
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap #COCOEvalCap_spice
except:
    print('Warning: coco-caption not available')

bad_endings = ['a','an','the','in','for','at','of','with','before','after','on','upon','near','to','is','are','am']
bad_endings += ['the']


def count_bad(sen):
    sen = sen.split(' ')
    if sen[-1] in bad_endings:
        return 1
    else:
        return 0


def getCOCO(dataset):
    if 'coco' in dataset:
        # annFile = 'coco-caption/annotations/captions_val2014.json'
        # annFile = 'coco-caption/annotations/captions_LN_val2014_norepeat.json' # load localized narratives for evaluation
        annFile = 'coco-caption/annotations/captions_LN_8kval.json' # load 8k LN validation set

    elif 'flickr30k' in dataset or 'f30k' in dataset or 'flk30k' in dataset:
        annFile = 'coco-caption/annotations/captions_flk30k_LN_test.json'
    elif 'ade20k' in dataset:
        annFile = 'coco-caption/annotations/captions_ade20k_LN_test.json'
    elif 'openimg' in dataset:
        annFile = 'coco-caption/annotations/captions_openimg_LN_test.json'
    print(annFile)
    return COCO(annFile)


cache_path = '/home/zihang/Research/Localized_Narratives/ImageCaptioning.pytorch/eval_results/zihang_transformer_LN_try804_openimg_twolayer_joint_cycle_b_val.json'
score_list = []
l = len(json.load(open(cache_path)))
size_per_split = 1000000
num_splits = (l//size_per_split) + (1 if (l%size_per_split)!=0 else 0)
for i in range(num_splits):
    coco = getCOCO('openimg')
    valids = coco.getImgIds()

    cocoRes = coco.loadRes(cache_path)#, split=i, size_per_split = size_per_split)
    cocoEval = COCOEvalCap(coco, cocoRes) #_spice
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    try:
        cocoEval.evaluate()
    except:
        print('this split fail: #', i)
        continue

    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score
        score_list.append(score)
    print(i, '-th current_split:', score, 'Overall ave:', sum(score_list) / len(score_list))
print(score_list)
print(sum(score_list) / len(score_list))

# # Add mean perplexity
# out['perplexity'] = mean_perplexity
# out['entropy'] = mean_entropy

imgToEval = cocoEval.imgToEval
for k in list(imgToEval.values())[0]['SPICE'].keys():
    if k != 'All':
        out['SPICE_' + k] = np.array([v['SPICE'][k]['f'] for v in imgToEval.values()])
        out['SPICE_' + k] = (out['SPICE_' + k][out['SPICE_' + k] == out['SPICE_' + k]]).mean()
for p in preds_filt:
    image_id, caption = p['image_id'], p['caption']
    imgToEval[image_id]['caption'] = caption

out['bad_count_rate'] = sum([count_bad(_['caption']) for _ in preds_filt]) / float(len(preds_filt))
outfile_path = os.path.join('eval_results/', model_id + '_' + split + '.json')
with open(outfile_path, 'w') as outfile:
    json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)
