import torch
import torch.nn.functional as F
from . import losses
from ..utils.rewards import init_scorer, get_self_critical_reward
import numpy as np
import random

class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        if opt.label_smoothing > 0:
            self.crit_caption = losses.LabelSmoothing(smoothing=opt.label_smoothing)
        else:
            self.crit_caption = losses.LanguageModelCriterion()
        self.rl_crit = losses.RewardCriterion()
        self.struc_crit = losses.StructureLosses(opt)

        # regression loss for trace generation
        self.crit_trace = torch.nn.L1Loss()

    def forward(self, fc_feats, att_feats, trace_feats, box_feats, labels, masks, att_masks, trace_masks, gts, gt_indices,
                sc_flag, struc_flag):
        opt = self.opt
        
        out = {}
        if struc_flag:
            if opt.structure_loss_weight < 1:
                lm_loss = self.crit(self.model(fc_feats, att_feats, labels[..., :-1], att_masks), labels[..., 1:], masks[..., 1:])
            else:
                lm_loss = torch.tensor(0).type_as(fc_feats)
            if opt.structure_loss_weight > 0:
                gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks,
                    opt={'sample_method':opt.train_sample_method,
                        'beam_size':opt.train_beam_size,
                        'output_logsoftmax': opt.struc_use_logsoftmax or opt.structure_loss_type == 'softmax_margin'\
                            or not 'margin' in opt.structure_loss_type,
                        'sample_n': opt.train_sample_n},
                    mode='sample')
                gts = [gts[_] for _ in gt_indices.tolist()]
                struc_loss = self.struc_crit(sample_logprobs, gen_result, gts)
            else:
                struc_loss = {'loss': torch.tensor(0).type_as(fc_feats),
                              'reward': torch.tensor(0).type_as(fc_feats)}
            loss = (1-opt.structure_loss_weight) * lm_loss + opt.structure_loss_weight * struc_loss['loss']
            out['lm_loss'] = lm_loss
            out['struc_loss'] = struc_loss['loss']
            out['reward'] = struc_loss['reward']
        elif not sc_flag:
            if self.opt.task == 'pred_both':
            # train generating both caption and trace
                caption_outputs_both, trace_outputs_both = self.model(fc_feats, att_feats, trace_feats, box_feats, labels[..., :-1],
                                                    att_masks, trace_masks, task='both')
                loss_mask = ((trace_masks != 0) * (trace_feats[:, :, 4] != 1)).unsqueeze(2)
                loss_both_trace = (torch.abs(trace_outputs_both[:, :, :4] - trace_feats[:, :, :4]) * loss_mask).sum() / (
                            loss_mask.sum() * 4)
                loss_both_caption = self.crit_caption(caption_outputs_both, labels[..., 1:], masks[..., 1:])
                loss_both = loss_both_caption + loss_both_trace # for baseline training
            if self.opt.task in ['caption', 'c_joint_t']:
                # for caption generation
                caption_outputs = self.model(fc_feats, att_feats, trace_feats, box_feats, labels[..., :-1],
                                                            att_masks, trace_masks, task='caption')
                loss_caption = self.crit_caption(caption_outputs, labels[..., 1:], masks[..., 1:])

            if self.opt.task in ['trace', 'c_joint_t']:
                # for trace generation - regression
                trace_outputs = self.model(fc_feats, att_feats, trace_feats, box_feats, labels[..., :-1],
                                                            att_masks, trace_masks, task='trace')
                loss_mask = ((trace_masks!=0) * (trace_feats[:,:,4]!=1)).unsqueeze(2) # for those words without labels ([0,0,1,1,1]), don't calculate the loss
                loss_trace = (torch.abs(trace_outputs[:,:,:4] -  trace_feats[:,:,:4]) * loss_mask).sum() / (loss_mask.sum() * 4)



            # # for cycle trace and caption
            # trace_outputs_both = trace_outputs_both.detach()
            # caption_outputs_cycle = self.model(fc_feats, att_feats, trace_outputs_both, box_feats, labels[..., :-1],
            #                              att_masks, trace_masks, task='caption')

            # caption_outputs_cycle_1 = torch.exp(caption_outputs) # get the logits before log (only after softmax)
            # trace_outputs_cycle_1 = self.model(fc_feats, att_feats, trace_feats, box_feats, caption_outputs_cycle_1,
            #                                             att_masks, trace_masks, task='cycle_trace')
            # loss_cycle_trace = (torch.abs(trace_outputs_cycle_1[:,:,:4] -  trace_feats[:,:,:4]) * loss_mask).sum() / (loss_mask.sum() * 4)
            #
            # trace_outputs_cycle_2 = trace_outputs
            # caption_outputs_cycle_2 = self.model(fc_feats, att_feats, trace_outputs_cycle_2, box_feats, labels[..., :-1],
            #                                             att_masks, trace_masks, task='caption')
            # loss_cycle_caption = self.crit_caption(caption_outputs_cycle_2, labels[..., 1:], masks[..., 1:])

            ################ random permute cycle loss ###################
            ### random permute trace within its segments
            # permute_trace_list = []
            # for i in range(trace_feats.shape[0]):
            #     tmp_gt_length = trace_masks[i].sum().long().item()
            #     tmp_trace = trace_feats[i, :tmp_gt_length]
            #     segment_list = []
            #     tmp_const = np.ceil(tmp_gt_length / 5).astype(int)
            #     for j in range(5):
            #         segment_list.append(tmp_trace[j * tmp_const: (j + 1) * tmp_const])
            #     random.shuffle(segment_list)
            #     tmp_permute_trace = torch.cat(segment_list, 0)
            #     if tmp_permute_trace.shape[0] < trace_masks.shape[1]:
            #         tmp_permute_trace = torch.cat([tmp_permute_trace,
            #                                        torch.zeros([trace_masks.shape[1]-tmp_permute_trace.shape[0], tmp_permute_trace.shape[1]]).to(trace_masks.device)])
            #     permute_trace_list.append(tmp_permute_trace)
            # permute_trace_feats = torch.stack(permute_trace_list, 0)
            #

            if self.opt.task == 'c_joint_t':
                #### random exchange trace within batch
                random_idx = np.arange(trace_feats.shape[0])
                np.random.shuffle(random_idx)
                rnd_trace_feats = trace_feats[random_idx]

                # construct the loss
                rnd_caption_outputs = self.model(fc_feats, att_feats, rnd_trace_feats, box_feats, labels[..., :-1],
                                                 att_masks, trace_masks, task='caption')
                caption_outputs_cycle_1 = torch.exp(rnd_caption_outputs)

                ## caption_outputs_cycle_1 = torch.exp(caption_outputs) # get the logits before log (only after softmax)
                trace_outputs_cycle_1 = self.model(fc_feats, att_feats, trace_feats, box_feats, caption_outputs_cycle_1,
                                                   att_masks, trace_masks, task='cycle_trace')
                loss_cycle_trace = (torch.abs(
                    trace_outputs_cycle_1[:, :, :4] - trace_feats[:, :, :4]) * loss_mask).sum() / (loss_mask.sum() * 4)

            if self.opt.task == 'pred_both':
                loss = loss_both
            elif self.opt.task == 'caption':
                loss = loss_caption
            elif self.opt.task == 'caption':
                loss = loss_trace
            elif self.opt.task == 'c_joint_t':
                loss =  loss_trace + 0.3 * (loss_caption) + 0.1 * (loss_cycle_trace)


        else:
            self.model.eval()
            with torch.no_grad():
                greedy_res, _ = self.model(fc_feats, att_feats, att_masks,
                    mode='sample',
                    opt={'sample_method': opt.sc_sample_method,
                         'beam_size': opt.sc_beam_size})
            self.model.train()
            gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks,
                    opt={'sample_method':opt.train_sample_method,
                        'beam_size':opt.train_beam_size,
                        'sample_n': opt.train_sample_n},
                    mode='sample')
            gts = [gts[_] for _ in gt_indices.tolist()]
            reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
            reward = torch.from_numpy(reward).to(sample_logprobs)
            loss = self.rl_crit(sample_logprobs, gen_result.data, reward)
            out['reward'] = reward[:,0].mean()
        out['loss'] = loss
        return out
