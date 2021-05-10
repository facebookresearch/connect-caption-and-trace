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
from . import misc as utils
from .local_optimal_transport import local_OT
from ..models import utils as utils_models

# load coco-caption if available
try:
    sys.path.append("coco-caption")
    from pycocotools.coco import COCO
    from pycocoevalcap.eval_show_control_tell import COCOEvalCap
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
        annFile = 'coco-caption/annotations/captions_val2014.json'
        # annFile = 'coco-caption/annotations/captions_LN_val2014_norepeat.json' # load localized narratives for evaluation
        # annFile = 'coco-caption/annotations/captions_LN_8kval.json' # load 8k LN validation set

    elif 'flickr30k' in dataset or 'f30k' in dataset:
        annFile = 'data/f30k_captions4eval.json'
    return COCO(annFile)


def language_eval(dataset, preds, preds_n, eval_kwargs, split):
    model_id = eval_kwargs['id']
    eval_oracle = eval_kwargs.get('eval_oracle', 0)
    
    # create output dictionary
    out = {}

    if len(preds_n) > 0:
        # vocab size and novel sentences
        if 'coco' in dataset:
            dataset_file = 'data/dataset_coco.json'
        elif 'flickr30k' in dataset or 'f30k' in dataset:
            dataset_file = 'data/dataset_flickr30k.json'
        training_sentences = set([' '.join(__['tokens']) for _ in json.load(open(dataset_file))['images'] if not _['split'] in ['val', 'test'] for __ in _['sentences']])
        generated_sentences = set([_['caption'] for _ in preds_n])
        novels = generated_sentences - training_sentences
        out['novel_sentences'] = float(len(novels)) / len(preds_n)
        tmp = [_.split() for _ in generated_sentences]
        words = []
        for _ in tmp:
            words += _
        out['vocab_size'] = len(set(words))

    # encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    cache_path = os.path.join('eval_results/', '.cache_'+ model_id + '_' + split + '.json')

    coco = getCOCO(dataset)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set
    preds_filt = [p for p in preds if p['image_id'] in valids]
    mean_perplexity = sum([_['perplexity'] for _ in preds_filt]) / len(preds_filt)
    mean_entropy = sum([_['entropy'] for _ in preds_filt]) / len(preds_filt)
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    for metric, score in cocoEval.eval.items():
        out[metric] = score
    # Add mean perplexity
    out['perplexity'] = mean_perplexity
    out['entropy'] = mean_entropy

    imgToEval = cocoEval.imgToEval
    for k in list(imgToEval.values())[0]['SPICE'].keys():
        if k != 'All':
            out['SPICE_'+k] = np.array([v['SPICE'][k]['f'] for v in imgToEval.values()])
            out['SPICE_'+k] = (out['SPICE_'+k][out['SPICE_'+k]==out['SPICE_'+k]]).mean()
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption

    if len(preds_n) > 0:
        from . import eval_multi
        cache_path_n = os.path.join('eval_results/', '.cache_'+ model_id + '_' + split + '_n.json')
        allspice = eval_multi.eval_allspice(dataset, preds_n, model_id, split)
        out.update(allspice['overall'])
        div_stats = eval_multi.eval_div_stats(dataset, preds_n, model_id, split)
        out.update(div_stats['overall'])
        if eval_oracle:
            oracle = eval_multi.eval_oracle(dataset, preds_n, model_id, split)
            out.update(oracle['overall'])
        else:
            oracle = None
        self_cider = eval_multi.eval_self_cider(dataset, preds_n, model_id, split)
        out.update(self_cider['overall'])
        with open(cache_path_n, 'w') as outfile:
            json.dump({'allspice': allspice, 'div_stats': div_stats, 'oracle': oracle, 'self_cider': self_cider}, outfile)
        
    out['bad_count_rate'] = sum([count_bad(_['caption']) for _ in preds_filt]) / float(len(preds_filt))
    outfile_path = os.path.join('eval_results/', model_id + '_' + split + '.json')
    with open(outfile_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out

def eval_split(model, crit, loader, task='caption', eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 0)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    sample_n = eval_kwargs.get('sample_n', 1)
    remove_bad_endings = eval_kwargs.get('remove_bad_endings', 0)
    os.environ["REMOVE_BAD_ENDINGS"] = str(remove_bad_endings) # Use this nasty way to make other code clean since it's a global configuration
    device = eval_kwargs.get('device', 'cuda')

    # assert task
    assert task in ['caption', 'trace', 'both', 'show']

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    n_predictions = [] # when sample_n > 1
    while True:
        data = loader.get_batch(split)
        # print('In eval_utils:', split)###zihang
        print(num_images)
        n = n + len(data['infos'])

        tmp = [data['fc_feats'], data['att_feats'], data['trace_feats'], data['box_feats'], data['labels'],
               data['masks'], data['att_masks'], data['trace_masks'], data['show_labels'], data['show_trace_feats'], data['show_trace_masks'], data['show_masks'], data['show_gate_labels']]
        tmp = [_.to(device) if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, trace_feats, box_feats, labels, masks, att_masks, \
            trace_masks, show_labels, show_trace_feats, show_trace_masks, show_masks, show_gate_labels = tmp
        if labels is not None and verbose_loss:
            # forward the model to get loss
            with torch.no_grad():
                if task == 'caption':
                    loss = crit(model(fc_feats, att_feats, trace_feats, box_feats, labels[..., :-1], att_masks, trace_masks, task=task), labels[..., 1:], masks[..., 1:]).item()
                elif task == 'show':
                    loss = crit(
                        model(fc_feats, att_feats, show_trace_feats, box_feats, show_labels[..., :-1], att_masks,
                              show_trace_masks, show_gate_labels, task=task), show_labels[..., 1:], show_masks[..., 1:]).item()
                elif task == 'both':
                    loss = crit(
                        model(fc_feats, att_feats, trace_feats, box_feats, labels[..., :-1], att_masks, trace_masks,
                              task=task)[0], labels[..., 1:], masks[..., 1:]).item()
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        with torch.no_grad():
            tmp_eval_kwargs = eval_kwargs.copy()
            tmp_eval_kwargs.update({'sample_n': 1})
            ### repeat att feats
            fc_feats, att_feats, att_masks, box_feats = utils_models.repeat_tensors(5,
                                                                   [fc_feats, att_feats, att_masks, box_feats]
                                                                   )
            #############################
            seq, seq_logprobs = model(fc_feats, att_feats, show_trace_feats, box_feats, att_masks, show_trace_masks, show_gate_labels, task, opt=tmp_eval_kwargs, mode='sample')
            seq = seq.data
            entropy = - (F.softmax(seq_logprobs, dim=2) * seq_logprobs).sum(2).sum(1) / ((seq>0).to(seq_logprobs).sum(1)+1)
            perplexity = - seq_logprobs.gather(2, seq.unsqueeze(2)).squeeze(2).sum(1) / ((seq>0).to(seq_logprobs).sum(1)+1)
            ### log which caption has no bounding box
            ids_no_box = (show_trace_feats[:, 0, 4] == 1).float()
    

        # Print beam search
        if beam_size > 1 and verbose_beam:
            for i in range(fc_feats.shape[0]):
                print('\n'.join([utils.decode_sequence(model.vocab, _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                print('--' * 10)
        sents = utils.decode_sequence(model.vocab, seq)
        for k, sent in enumerate(sents):
            # entry = {'image_id': data['infos'][k]['id'], 'caption': sent, 'perplexity': perplexity[k].item(), 'entropy': entropy[k].item()}
            # entry to evaluate show-control-tell: seperate the 5 predictions per image
       
            if ids_no_box[k]==1:
                continue
            entry = {'image_id': data['infos'][k//5]['id'] + 1000000 * (k%5), 'caption': sent, 'perplexity': perplexity[k].item(),
                     'entropy': entropy[k].item()}

            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                print(cmd)
                os.system(cmd)

            if verbose:
                print('image %s: %s' %(entry['image_id'], entry['caption']))

        if sample_n > 1:
            eval_split_n(model, n_predictions, [fc_feats, att_feats, trace_feats, box_feats, att_masks, trace_masks, data], eval_kwargs)
        
        # ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        # print('ix1', ix1)###zihang
        if num_images != -1:
            ix1 = min(ix1, num_images)
        else:
            num_images = ix1
        # print('len:', len(predictions), n, ix1, split, num_images)  ###zihang
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(n, ix1, loss))

        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if len(n_predictions) > 0 and 'perplexity' in n_predictions[0]:
        n_predictions = sorted(n_predictions, key=lambda x: x['perplexity'])
    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    torch.save((predictions, n_predictions), os.path.join('eval_results/', '.saved_pred_'+ eval_kwargs['id'] + '_' + split + '.pth'))
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, n_predictions, eval_kwargs, split)

    # Switch back to training mode
    model.train()
    return loss_sum/loss_evals, predictions, lang_stats


# Only run when sample_n > 0
def eval_split_n(model, n_predictions, input_data, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    beam_size = eval_kwargs.get('beam_size', 1)
    sample_n = eval_kwargs.get('sample_n', 1)
    sample_n_method = eval_kwargs.get('sample_n_method', 'sample')

    fc_feats, att_feats, trace_feats, box_feats, att_masks, trace_masks, data = input_data

    tmp_eval_kwargs = eval_kwargs.copy()
    if sample_n_method == 'bs':
        # case 1 sample_n == beam size
        tmp_eval_kwargs.update({'sample_n': 1, 'beam_size': sample_n, 'group_size': 1}) # randomness from softmax
        with torch.no_grad():
            model(fc_feats, att_feats, trace_feats, box_feats, att_masks, trace_masks, opt=tmp_eval_kwargs, mode='sample')
        for k in range(fc_feats.shape[0]):
            _sents = utils.decode_sequence(model.vocab, torch.stack([model.done_beams[k][_]['seq'] for _ in range(sample_n)]))
            for sent in _sents:
                entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
                n_predictions.append(entry)
    # case 2 sample / gumbel / topk sampling/ nucleus sampling
    elif sample_n_method == 'sample' or \
            sample_n_method == 'gumbel' or \
            sample_n_method.startswith('top'):
        tmp_eval_kwargs.update({'sample_n': sample_n, 'sample_method': sample_n_method, 'beam_size': 1}) # randomness from sample
        with torch.no_grad():
            _seq, _sampleLogprobs = model(fc_feats, att_feats, trace_feats, box_feats, att_masks, trace_masks, opt=tmp_eval_kwargs, mode='sample')
        _sents = utils.decode_sequence(model.vocab, _seq)
        _perplexity = - _sampleLogprobs.gather(2, _seq.unsqueeze(2)).squeeze(2).sum(1) / ((_seq>0).to(_sampleLogprobs).sum(1)+1)
        for k, sent in enumerate(_sents):
            entry = {'image_id': data['infos'][k // sample_n]['id'], 'caption': sent, 'perplexity': _perplexity[k].item()}
            n_predictions.append(entry)
    elif sample_n_method == 'dbs':
        # Use diverse beam search
        tmp_eval_kwargs.update({'beam_size': sample_n * beam_size, 'group_size': sample_n}) # randomness from softmax
        with torch.no_grad():
            model(fc_feats, att_feats, trace_feats, box_feats, att_masks, trace_masks, opt=tmp_eval_kwargs, mode='sample')
        for k in range(loader.batch_size):
            _sents = utils.decode_sequence(model.vocab, torch.stack([model.done_beams[k][_]['seq'] for _ in range(0, sample_n*beam_size, beam_size)]))
            for sent in _sents:
                entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
                n_predictions.append(entry)
    else:
        tmp_eval_kwargs.update({'sample_method': sample_n_method[1:], 'group_size': sample_n, 'beam_size':1}) # randomness from softmax
        with torch.no_grad():
            _seq, _sampleLogprobs = model(fc_feats, att_feats, trace_feats, box_feats, att_masks, trace_masks, opt=tmp_eval_kwargs, mode='sample')
        _sents = utils.decode_sequence(model.vocab, _seq)
        for k, sent in enumerate(_sents):
            entry = {'image_id': data['infos'][k // sample_n]['id'], 'caption': sent}
            n_predictions.append(entry)
    if verbose:
        for entry in sorted(n_predictions[-fc_feats.shape[0] * sample_n:], key=lambda x: x['image_id']):
            print('image %s: %s' %(entry['image_id'], entry['caption']))


def eval_trace_generation(model, crit, loader, eval_kwargs={}):
    model.eval()
    count = 0
    split = 'val'
    loader.reset_iterator(split)
    device = 'cuda'
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    while True:
        data = loader.get_batch(split)
        # print(data['infos'][0]['id'])
        ix1 = data['bounds']['it_max']
        # print('ix1', ix1)###zihang
        if num_images != -1:
            ix1 = min(ix1, num_images)
        else:
            num_images = ix1
        tmp = [data['fc_feats'], data['att_feats'], data['trace_feats'], data['box_feats'], data['labels'],
               data['masks'], data['att_masks'], data['trace_masks']]
        tmp = [_.to(device) if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, trace_feats, box_feats, labels, masks, att_masks, trace_masks = tmp
        loss_ce_list = []
        gt_prev_acc_list = []
        loss_list = []
        loss_prev_gt_list = []
        acc_list = []
        with torch.no_grad():
            use_local_OT = False
            # get the loss for l1-loss and classification accuracy
            tmp_trace_feats = trace_feats.clone()
            # trace_class_label = trace_feats[:,:,5] - 1
            # pred_class_label = torch.zeros_like(trace_class_label).to(trace_class_label.device)

            # prev_gt_correct
            prev_gt_out = model(fc_feats, att_feats, tmp_trace_feats, box_feats, labels[..., :-1], att_masks, trace_masks, 'trace')
            prev_gt_out = prev_gt_out * trace_masks.unsqueeze(2)
            # print(prev_gt_out[0, :, :5])

            loss_prev_gt_mask = ((trace_masks != 0) * (trace_feats[:, :, 4] != 1)).unsqueeze(2)
            loss_prev_gt =  ((torch.abs(prev_gt_out[:, :, :4] - trace_feats[:, :, :4]) * loss_prev_gt_mask).sum() / (
                        loss_prev_gt_mask.sum() * 4)).item()
            loss_prev_gt_list.append(loss_prev_gt)

            for i in range(trace_feats.shape[1]):
                # for regression
                curr_out = model(fc_feats, att_feats, tmp_trace_feats, box_feats, labels[..., :-1], att_masks, trace_masks, 'trace')[:, i]
                curr_out[:, 4] = (curr_out[:, 2] - curr_out[:, 0]) * (curr_out[:, 3] - curr_out[:, 1])
                tmp_trace_feats[:, i, :5] = curr_out
                # curr_out = model(fc_feats, att_feats, tmp_trace_feats, box_feats, labels[..., :-1], att_masks, trace_masks) # for non-iteratively
                # break # for non-iteratively
            # tmp_trace_feats = curr_out # for non-iteratively

            # ### save for visualization  # for visualization of trace_generation
            # vis_img_id = data['infos'][0]['id']
            # np.save('./vis/trace_generation_2/pred_trace_'+str(vis_img_id), tmp_trace_feats[:,:,:4].detach().cpu().numpy())
            # np.save('./vis/trace_generation_2/gt_trace_' + str(vis_img_id), trace_feats[:,:,:4].detach().cpu().numpy())
            # print(vis_img_id, crit(tmp_trace_feats[:,:,:4], trace_feats[:,:,:4]).item(), trace_feats.shape)
            # with open('./vis/trace_generation_2/info.txt', 'a') as f:
            #     f.write('img_id:%d, l1-loss: %f\n'%(vis_img_id,(crit(tmp_trace_feats[:,:,:4], trace_feats[:,:,:4]) * trace_masks.shape[0]*trace_masks.shape[1] / (trace_masks!=0).sum()).item()))
            #     f.close()
            # ############################


            # tmp_trace_feats = tmp_trace_feats * trace_masks.unsqueeze(2)
            loss_mask = ((trace_masks != 0) * (trace_feats[:, :, 4] != 1)).unsqueeze(2) #
            if use_local_OT:
                D = torch.abs(tmp_trace_feats[:, :, :4].unsqueeze(2) - trace_feats[:, :, :4].unsqueeze(1)).mean(dim=-1)
                T = local_OT(D).to(tmp_trace_feats.device)
                loss = ((torch.abs(torch.matmul(tmp_trace_feats[:, :, :4].transpose(1, 2), T).transpose(1, 2) -
                                  trace_feats[:, :, :4]) * loss_mask).sum() / (loss_mask.sum() * 4)).item()
                print('loss', loss, 'loss_orig', (
                            (torch.abs(tmp_trace_feats[:, :, :4] - trace_feats[:, :, :4]) * loss_mask).sum() / (
                                loss_mask.sum() * 4)).item())
            else:
                loss = ((torch.abs(tmp_trace_feats[:, :, :4] - trace_feats[:, :, :4]) * loss_mask).sum() / (loss_mask.sum() * 4)).item()

            # loss = (crit(tmp_trace_feats[:,:,:4], trace_feats[:,:,:4]) * trace_masks.shape[0]*trace_masks.shape[1] / (trace_masks!=0).sum()).item()
            loss_list.append(loss)
        count += att_feats.shape[0]
        print('Validation evaluation(%d/%d):'%(count, num_images), 'l1-loss:', loss, '; prev_gt_loss:', loss_prev_gt)
        if count >= num_images: ### currently use 5000 in validation set
            break

    val_loss = np.mean(np.array(loss_list))
    val_loss_prev_gt = np.mean(np.array(loss_prev_gt_list))
    print('Validation evaluation: loss', 'l1-loss:', val_loss, '; prev_gt_loss:', val_loss_prev_gt)
    model.train()
    return val_loss

def eval_trace_generation_classification(model, crit, loader, eval_kwargs={}):
    model.eval()
    count = 0
    split = 'val'
    loader.reset_iterator(split)
    device = 'cuda'
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    while True:
        data = loader.get_batch(split)
        ix1 = data['bounds']['it_max']
        # print('ix1', ix1)###zihang
        if num_images != -1:
            ix1 = min(ix1, num_images)
        else:
            num_images = ix1
        tmp = [data['fc_feats'], data['att_feats'], data['trace_feats'], data['box_feats'], data['labels'],
               data['masks'], data['att_masks'], data['trace_masks']]
        tmp = [_.to(device) if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, trace_feats, box_feats, labels, masks, att_masks, trace_masks = tmp
        loss_ce_list = []
        gt_prev_acc_list = []
        loss_list = []
        acc_list = []
        with torch.no_grad():
            # get the loss in terms of cross-entropy
            model_outputs = model(fc_feats, att_feats, trace_feats, box_feats, labels[..., :-1], att_masks,
                                  trace_masks, 'trace')
            model_outputs = F.log_softmax(model_outputs, dim=-1)
            model_outputs = model_outputs.view(-1, model_outputs.shape[2])
            trace_class_label = trace_feats[:, :, 5] - 1
            trace_class_label = trace_class_label.view(-1).long()
            loss_ce = F.nll_loss(model_outputs, trace_class_label, ignore_index=-1).item()
            loss_ce_list.append(loss_ce)
            gt_prev_acc = (((model_outputs.argmax(dim=1) == trace_class_label)*(trace_class_label!=-1).float()).sum() / \
                          (trace_class_label != -1).float().sum()).item()
            gt_prev_acc_list.append(gt_prev_acc)

            # get the loss for l1-loss and classification accuracy
            tmp_trace_feats = trace_feats
            trace_class_label = trace_feats[:,:,5] - 1
            pred_class_label = torch.zeros_like(trace_class_label).to(trace_class_label.device)

            for i in range(trace_feats.shape[1]):
                # for regression
                # curr_out = model(fc_feats, att_feats, tmp_trace_feats, box_feats, labels[..., :-1], att_masks, trace_masks)[:, i]
                # for classification
                curr_out = model(fc_feats, att_feats, tmp_trace_feats, box_feats, labels[..., :-1], att_masks, trace_masks, 'trace')[:,i]
                curr_out = curr_out.argmax(dim=-1)
                pred_class_label[:, i] = curr_out
                curr_out = box_feats[np.arange(box_feats.shape[0]), curr_out]
                tmp_trace_feats[:, i, :5] = curr_out
            print('prev_gt_class_label', model_outputs.argmax(dim=1).view(pred_class_label.shape)[0])
            print('pred_class_label',pred_class_label[0])
            classification_acc = ((pred_class_label == trace_class_label) * trace_masks).sum() / trace_masks.sum()
            acc_list.append(classification_acc.item())
            tmp_trace_feats = tmp_trace_feats * trace_masks.unsqueeze(2)
            loss = crit(tmp_trace_feats[:,:,:4], trace_feats[:,:,:4]).item()
            loss_list.append(loss)
        count += att_feats.shape[0]
        print('Validation evaluation(%d/%d):'%(count, num_images), 'l1-loss:', loss, '; loss_ce:', loss_ce, '; classification-acc:', classification_acc.item(), '; gt_prev_acc:', gt_prev_acc)
        if count >= num_images: ### currently use 5000 in validation set
            break
    val_loss_ce = np.mean(np.array(loss_ce_list))
    val_gt_prev_acc = np.mean(np.array(gt_prev_acc_list))
    val_loss = np.mean(np.array(loss_list))
    val_acc = np.mean(np.array(acc_list))
    print('Validation evaluation: loss', 'l1-loss:', val_loss, '; loss_ce:', val_loss_ce, '; classification-acc:', val_acc, '; gt_prev_acc:', val_gt_prev_acc)
    model.train()
    return val_loss
