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

# load coco-caption if available
try:
    sys.path.append("coco-caption")
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
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
        annFile = 'coco-caption/annotations/captions_coco_LN_test.json'
    elif 'flickr30k' in dataset or 'f30k' in dataset or 'flk30k' in dataset:
        annFile = 'coco-caption/annotations/captions_flk30k_LN_test.json'
    elif 'ade20k' in dataset:
        annFile = 'coco-caption/annotations/captions_ade20k_LN_test.json'
    elif 'openimg' in dataset:
        annFile = 'coco-caption/annotations/captions_openimg_LN_test.json'
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
    assert task in ['caption', 'trace', 'both']

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    n_predictions = [] # when sample_n > 1
    trace_cost = []
    while True:
        data = loader.get_batch(split)
        n = n + len(data['infos'])

        tmp = [data['fc_feats'], data['att_feats'], data['trace_feats'], data['box_feats'], data['labels'], data['masks'], data['att_masks'], data['trace_masks']]
        tmp = [_.to(device) if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, trace_feats, box_feats, labels, masks, att_masks, trace_masks = tmp
        if labels is not None and verbose_loss:
            # forward the model to get loss
            with torch.no_grad():
                if task == 'caption':
                    loss = crit(model(fc_feats, att_feats, trace_feats, box_feats, labels[..., :-1], att_masks, trace_masks, task=task), labels[..., 1:], masks[..., 1:]).item()
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
            if task == 'both':
                seq, seq_logprobs, trace_predicted = model(fc_feats, att_feats, trace_feats, box_feats, att_masks, trace_masks,
                                          task=task, opt=tmp_eval_kwargs, mode='sample')
            else:
                try:
                    seq, seq_logprobs = model(fc_feats, att_feats, trace_feats, box_feats, att_masks, trace_masks, task=task, opt=tmp_eval_kwargs, mode='sample')
                except:
                    print('evaluation meet error')
                    continue
            seq = seq.data
            entropy = - (F.softmax(seq_logprobs, dim=2) * seq_logprobs).sum(2).sum(1) / ((seq>0).to(seq_logprobs).sum(1)+1)
            perplexity = - seq_logprobs.gather(2, seq.unsqueeze(2)).squeeze(2).sum(1) / ((seq>0).to(seq_logprobs).sum(1)+1)

        if task == 'both':
            ### compute the loss for trace
            for k in range(trace_predicted.shape[0]):
                tmp_gt_length = trace_masks[k].sum().long()
                tmp_gt_trace = trace_feats[k, :tmp_gt_length]
                tmp_pred_length = (seq[k]>0).sum().long()
                tmp_pred_trace = trace_predicted[k, :tmp_pred_length]

                # choose only boxes not [0,0,1,1,1] in the ground truth
                nonzero_idx = torch.nonzero(tmp_gt_trace[:, 4] != 1).squeeze()
                tmp_gt_trace = tmp_gt_trace[nonzero_idx]
                if len(tmp_gt_trace.shape) < 2:  # if there is only one chosen box in this trace
                    tmp_gt_trace = tmp_gt_trace.unsqueeze(0)
                tmp_gt_trace = tmp_gt_trace.unsqueeze(0)
                tmp_pred_trace = tmp_pred_trace.unsqueeze(0)

                if tmp_pred_trace.shape[1] <= tmp_gt_trace.shape[1]:
                    tmp_trace1 = tmp_pred_trace
                    tmp_trace2 = tmp_gt_trace
                else:
                    tmp_trace1 = tmp_gt_trace
                    tmp_trace2 = tmp_pred_trace
                # processing in terms of segments of length 20
                seg_loss_list = []
                for seg_idx in range(np.ceil(tmp_trace1.shape[1] / 20).astype(int)):
                    tmp_const = 20. * tmp_trace2.shape[1] / tmp_trace1.shape[1]
                    seg_tmp_trace1 = tmp_trace1[:, seg_idx * 20:(seg_idx + 1) * 20, :4]
                    seg_tmp_trace2 = tmp_trace2[:, np.floor(seg_idx * tmp_const).astype(int): np.ceil(
                        (seg_idx + 1) * tmp_const).astype(int), :4]
                    D = torch.abs(seg_tmp_trace1.unsqueeze(2) - seg_tmp_trace2.unsqueeze(1)).mean(dim=-1)
                    seg_tmp_T = local_OT(D, window = 0)
                    seg_tmp_cost = (seg_tmp_T * D).sum() / seg_tmp_trace1.shape[1]
                    if not torch.isnan(seg_tmp_cost):
                        seg_loss_list.append(seg_tmp_cost.item())
                tmp_cost = np.mean(np.array(seg_loss_list))
                if not np.isnan(tmp_cost):
                    trace_cost.append(tmp_cost)
                print('trace LBM distance:', tmp_cost)

        # Print beam search
        if beam_size > 1 and verbose_beam:
            for i in range(fc_feats.shape[0]):
                print('\n'.join([utils.decode_sequence(model.vocab, _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                print('--' * 10)
        sents = utils.decode_sequence(model.vocab, seq)
        print('both trace running ave LBM loss :', np.mean(np.array(trace_cost)))

        # ### save for visualization  # for visualization of trace_generation
        # for i in range(len(sents)):
        #     vis_img_id = data['infos'][i]['id']
        #     with open('./vis/both_generation_supplement/pred_caption/pred_caption_' + str(vis_img_id)+'.txt', 'w') as f:
        #         f.write(sents[i])
        #     np.save('./vis/both_generation_supplement/pred_trace/pred_trace_' + str(vis_img_id),
        #             trace_predicted[i, :, :4].detach().cpu().numpy())
        #     print(vis_img_id, trace_feats.shape)
        #     with open('./vis/both_generation_supplement/info.txt', 'a') as f:
        #         f.write('img_id:%d\n' %vis_img_id)
        #         f.close()
        # ############################

        # ### save for visualization  # for visualization of caption_generation
        # for i in range(len(sents)):
        #     vis_img_id = data['infos'][i]['id']
        #     tmp_dir = './vis/caption_generation_' + eval_kwargs['dataset_choice']
        #     if not os.path.exists(tmp_dir):
        #         os.makedirs(tmp_dir)
        #         os.makedirs(tmp_dir + '/pred_caption')
        #         os.makedirs(tmp_dir + '/gt_trace')
        #     with open('./vis/caption_generation_'+ eval_kwargs['dataset_choice'] +'/pred_caption/pred_caption_' + str(vis_img_id) + '.txt',
        #               'w') as f:
        #         f.write(sents[i])
        #     np.save('./vis/caption_generation_'+ eval_kwargs['dataset_choice'] +'/gt_trace/gt_trace_' + str(vis_img_id),
        #             trace_feats[i, :, :4].detach().cpu().numpy())
        #     print(vis_img_id, trace_feats.shape)
        #     with open('./vis/caption_generation_'+ eval_kwargs['dataset_choice'] +'/info.txt', 'a') as f:
        #         f.write('img_id:%s\n' % str(vis_img_id))
        #         f.close()
        # ############################


        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent, 'perplexity': perplexity[k].item(), 'entropy': entropy[k].item()}
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

        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        else:
            num_images = ix1

        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(n, ix1, loss))

        if num_images >= 0 and n >= num_images:
            break

    if task == 'both':
        print('both trace total LBM loss:', np.mean(np.array(trace_cost)))

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


def eval_trace_generation(model, crit, loader, window_size=0, eval_kwargs={}):
    model.eval()
    count = 0
    split = 'val'
    loader.reset_iterator(split)
    device = 'cuda'
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    loss_list = []

    while True:
        data = loader.get_batch(split)
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        else:
            num_images = ix1
        tmp = [data['fc_feats'], data['att_feats'], data['trace_feats'], data['box_feats'], data['labels'],
               data['masks'], data['att_masks'], data['trace_masks']]
        tmp = [_.to(device) if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, trace_feats, box_feats, labels, masks, att_masks, trace_masks = tmp

        with torch.no_grad():
            # get the loss for l1-loss and classification accuracy
            tmp_trace_feats = trace_feats.clone()

            for i in range(trace_feats.shape[1]):
                # for regression
                curr_out = model(fc_feats, att_feats, tmp_trace_feats, box_feats, labels[..., :-1], att_masks, trace_masks, 'trace')[:, i]
                curr_out[:, 4] = (curr_out[:, 2] - curr_out[:, 0]) * (curr_out[:, 3] - curr_out[:, 1])
                tmp_trace_feats[:, i, :5] = curr_out

            # # ### save for visualization  # for visualization of trace_generation
            # sents = utils.decode_sequence(model.vocab, labels[:, 0, 1:])
            # print(sents)
            # loss_mask = ((trace_masks != 0) * (trace_feats[:, :, 4] != 1)).unsqueeze(2)  #
            # vis_img_id = data['infos'][0]['id']
            # tmp_dir = './vis/trace_generation_' + eval_kwargs['dataset_choice']
            # if not os.path.exists(tmp_dir):
            #     os.makedirs(tmp_dir)
            #     os.makedirs(tmp_dir + '/pred_trace')
            #     os.makedirs(tmp_dir + '/gt_trace')
            #     os.makedirs(tmp_dir + '/gt_caption')
            # with open(tmp_dir + '/gt_caption/' + str(vis_img_id)+'.txt', 'w') as f:
            #     f.write(sents[0])
            # # np.save('./vis/trace_generation_11_14/pred_caption_' + str(vis_img_id),
            # #         labels[..., 1:].detach().cpu().numpy())
            # np.save(tmp_dir + '/pred_trace/' +str(vis_img_id), tmp_trace_feats[:,:,:4].detach().cpu().numpy())
            # np.save(tmp_dir + '/gt_trace/' + str(vis_img_id), trace_feats[:,:,:4].detach().cpu().numpy())
            # print(vis_img_id, crit(tmp_trace_feats[:,:,:4], trace_feats[:,:,:4]).item(), trace_feats.shape)
            # with open(tmp_dir + '/info.txt', 'a') as f:
            #     f.write('img_id:%s, l1-loss: %f\n'%(str(vis_img_id),((torch.abs(tmp_trace_feats[:, :, :4] - trace_feats[:, :, :4]) * loss_mask).sum() / (loss_mask.sum() * 4)).item()))
            #     f.close()
            # # ############################

            use_local_OT = True #
            loss_mask = ((trace_masks != 0) * (trace_feats[:, :, 4] != 1)).unsqueeze(2) #
            if use_local_OT:
                batch_loss_list = []
                for idx_trace in range(trace_feats.shape[0]):
                    tmp_gt_length = trace_masks[idx_trace].sum().long()
                    single_tmp_trace_feats = tmp_trace_feats[idx_trace, :tmp_gt_length]
                    single_trace_feats = trace_feats[idx_trace, :tmp_gt_length]
                    # choose only boxes not [0,0,1,1,1] in the ground truth
                    nonzero_idx = torch.nonzero(single_trace_feats[:,4]!=1).squeeze()
                    single_trace_feats = single_trace_feats[nonzero_idx]
                    if len(single_trace_feats.shape) < 2: # if there is only one chosen box in this trace
                        single_trace_feats = single_trace_feats.unsqueeze(0)
                    single_tmp_trace_feats = single_tmp_trace_feats.unsqueeze(0)
                    single_trace_feats = single_trace_feats.unsqueeze(0)
                    if single_tmp_trace_feats.shape[1] <= single_trace_feats.shape[1]:
                        tmp_trace1 = single_tmp_trace_feats
                        tmp_trace2 = single_trace_feats
                    else:
                        tmp_trace1 = single_trace_feats
                        tmp_trace2 = single_tmp_trace_feats
                    # processing in terms of segments of length 20
                    seg_loss_list =  []

                    for seg_idx in range(np.ceil(tmp_trace1.shape[1]/20).astype(int)):
                        tmp_const = 20. * tmp_trace2.shape[1] / tmp_trace1.shape[1]
                        seg_tmp_trace1 = tmp_trace1[:, seg_idx*20:(seg_idx+1)*20, :4]
                        seg_tmp_trace2 = tmp_trace2[:, np.floor(seg_idx*tmp_const).astype(int) : np.ceil((seg_idx+1)*tmp_const).astype(int) , :4]
                        D = torch.abs(seg_tmp_trace1.unsqueeze(2) - seg_tmp_trace2.unsqueeze(1)).mean(dim=-1)
                        seg_tmp_T = local_OT(D, window = window_size)
                        seg_tmp_cost = (seg_tmp_T * D ).sum() / seg_tmp_trace1.shape[1]
                        if not torch.isnan(seg_tmp_cost):
                            seg_loss_list.append(seg_tmp_cost.item())

                    if len(seg_loss_list) != 0:
                        batch_loss_list.append(np.mean(np.array(seg_loss_list)))
                loss = np.mean(np.array(batch_loss_list))

                # loss_orig = ((torch.abs(tmp_trace_feats[:, :, :4] - trace_feats[:, :, :4]) * loss_mask).sum() / (loss_mask.sum() * 4)).item()

            else:
                loss = ((torch.abs(tmp_trace_feats[:, :, :4] - trace_feats[:, :, :4]) * loss_mask).sum() / (loss_mask.sum() * 4)).item()
                # loss_orig = loss

            if not np.isnan(loss):
                loss_list.append(loss)
            # loss_orig_list.append(loss_orig)

        print('Running ave l1 loss:', np.mean(np.array(loss_list))) #, np.mean(np.array(loss_orig_list)))
        count += att_feats.shape[0]
        print('Validation evaluation(%d/%d):'%(count, num_images), 'l1-loss:', loss)
        if count >= num_images: ### currently use 5000 in validation set
            break

    val_loss = np.mean(np.array(loss_list))
    print('Validation evaluation:', 'l1-loss:', val_loss)
    model.train()
    return val_loss