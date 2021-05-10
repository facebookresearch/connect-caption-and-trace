import torch
import torch.nn.functional as F
from . import losses
from ..utils.rewards import init_scorer, get_self_critical_reward

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
        self.show_gate_crit = torch.nn.CrossEntropyLoss()

        # regression loss for trace generation
        self.crit_trace = torch.nn.L1Loss()

    def forward(self, fc_feats, att_feats, trace_feats, box_feats, labels, masks, att_masks, trace_masks,
                show_labels, show_trace_feats, show_trace_masks, show_masks, show_gate_labels,
                gts, gt_indices, sc_flag, struc_flag):
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
            # train generating both caption and trace
            # caption_outputs_both, trace_outputs_both = self.model(fc_feats, att_feats, trace_feats, box_feats, labels[..., :-1],
            #                                     att_masks, trace_masks, task='both')
            # loss_mask = ((trace_masks != 0) * (trace_feats[:, :, 4] != 1)).unsqueeze(2)
            # loss_both_trace = (torch.abs(trace_outputs_both[:, :, :4] - trace_feats[:, :, :4]) * loss_mask).sum() / (
            #             loss_mask.sum() * 4)
            # loss_both_caption = self.crit_caption(caption_outputs_both, labels[..., 1:], masks[..., 1:])
            # loss_both = loss_both_caption + loss_both_trace
            # # # #
            # # # # for caption generation
            # # caption_outputs = self.model(fc_feats, att_feats, trace_feats, box_feats, labels[..., :-1],
            # #                                             att_masks, trace_masks, task='caption')
            # # loss_caption = self.crit_caption(caption_outputs, labels[..., 1:], masks[..., 1:])
            # #
            # # # for trace generation - regression
            # trace_outputs = self.model(fc_feats, att_feats, trace_feats, box_feats, labels[..., :-1],
            #                                             att_masks, trace_masks, task='trace')[:, :-1]
            # loss_mask = ((trace_masks!=0) * (trace_feats[:,:,4]!=1)).unsqueeze(2) # for those words without labels ([0,0,1,1,1]), don't calculate the loss
            # loss_trace = (torch.abs(trace_outputs[:,:,:4] -  trace_feats[:,:,:4]) * loss_mask).sum() / (loss_mask.sum() * 4)
            #
            # for coco-caption training
            ### inference to get with coco trace
            with torch.no_grad():
                tmp_trace_feats = show_trace_feats[:, :1]
                for i in range(show_labels.shape[2]-2):
                    # for regression
                    tmp_trace_feats_input = torch.cat(
                        [tmp_trace_feats, torch.zeros(tmp_trace_feats.shape[0], 1, tmp_trace_feats.shape[2]).to(tmp_trace_feats.device)], 1)
                    _, curr_out = self.model(fc_feats, att_feats, tmp_trace_feats_input, box_feats,
                                                    show_labels[..., :-1].squeeze(1),
                                                    att_masks, show_masks.squeeze(1)[:, :tmp_trace_feats_input.shape[1]], task='both')
                    curr_out = curr_out[:, i]
                    curr_out[:, 4] = (curr_out[:, 2] - curr_out[:, 0]) * (curr_out[:, 3] - curr_out[:, 1])
                    if i == 0:
                        tmp_trace_feats = curr_out.unsqueeze(1)
                    else:
                        tmp_trace_feats = torch.cat([tmp_trace_feats, curr_out.unsqueeze(1)], 1)

            coco_trace_outputs = tmp_trace_feats.detach()
            coco_caption_outputs, coco_trace_outputs_both = self.model(fc_feats, att_feats, coco_trace_outputs, box_feats,
                                                                  show_labels[..., :-1].squeeze(1),
                                                                  att_masks, show_masks.squeeze(1)[:, :coco_trace_outputs.shape[1]], task='both')
            loss_coco_caption = self.crit_caption(coco_caption_outputs, show_labels[..., 1:], show_masks[..., 1:])

            # # for coco-caption-baseline
            # baseline_caption_outputs = self.model(fc_feats, att_feats, show_trace_feats, box_feats, show_labels[..., :-1],
            #                                             att_masks, show_masks, task='caption')
            # loss_coco_caption_baseline = self.crit_caption(baseline_caption_outputs, show_labels[..., 1:], show_masks[..., 1:])

            # # # for show-control-tell
            # show_caption_outputs, show_gate_outputs = self.model(fc_feats, att_feats, show_trace_feats, box_feats, show_labels[..., :-1],
            #                              att_masks, show_trace_masks, show_gate_labels=show_gate_labels, task='show')
            # loss_show_caption = self.crit_caption(show_caption_outputs, show_labels[..., 1:], show_masks[..., 1:])
            # loss_show_gate = self.show_gate_crit(show_gate_outputs.reshape(-1, show_gate_outputs.shape[-1]),
            #                                      show_gate_labels[..., 1:].reshape(-1))

            # # # for cycle trace and caption
            # # trace_outputs_both = trace_outputs_both.detach()
            # # caption_outputs_cycle = self.model(fc_feats, att_feats, trace_outputs_both, box_feats, labels[..., :-1],
            # #                              att_masks, trace_masks, task='caption')
            #
            # caption_outputs_cycle_1 = torch.exp(caption_outputs) # get the logits before log (only after softmax)
            # trace_outputs_cycle_1 = self.model(fc_feats, att_feats, trace_feats, box_feats, caption_outputs_cycle_1,
            #                                             att_masks, trace_masks, task='cycle_trace')
            # loss_cycle_trace = (torch.abs(trace_outputs_cycle_1[:,:,:4] -  trace_feats[:,:,:4]) * loss_mask).sum() / (loss_mask.sum() * 4)
            #
            # trace_outputs_cycle_2 = trace_outputs
            # caption_outputs_cycle_2 = self.model(fc_feats, att_feats, trace_outputs_cycle_2, box_feats, labels[..., :-1],
            #                                             att_masks, trace_masks, task='caption')
            # loss_cycle_caption = self.crit_caption(caption_outputs_cycle_2, labels[..., 1:], masks[..., 1:])


            # sum the loss of caption and trace generation
            loss = loss_coco_caption #loss_both + loss_trace #+ loss_caption # loss_coco_caption  # loss_caption + loss_trace + loss_both # + (loss_cycle_caption + loss_cycle_trace) * 0.5 + loss_caption + loss_trace

            # for trace generation - classification
            # model_outputs = self.model(fc_feats, att_feats, trace_feats, box_feats, labels[..., :-1], att_masks, trace_masks)
            # model_outputs = F.log_softmax(model_outputs, dim=-1)
            # model_outputs = model_outputs.view(-1, model_outputs.shape[2])
            # trace_class_label = trace_feats[:,:,5] - 1
            # trace_class_label = trace_class_label.view(-1).long()
            # loss = F.nll_loss(model_outputs, trace_class_label, ignore_index=-1)
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
