import torch
import torch.nn.functional as F
from . import losses
from ..utils.rewards import init_scorer, get_self_critical_reward
from ..utils.local_optimal_transport import local_OT


class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        # if opt.label_smoothing > 0:
        #     self.crit = losses.LabelSmoothing(smoothing=opt.label_smoothing)
        # else:
        #     self.crit = losses.LanguageModelCriterion()
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
            # for caption generation
            # loss = self.crit(self.model(fc_feats, att_feats, trace_feats, box_feats, labels[..., :-1], att_masks, trace_masks), labels[..., 1:], masks[..., 1:])

            # for trace generation - regression
            # outputs = self.model(fc_feats, att_feats, trace_feats, box_feats, labels[..., :-1], att_masks, trace_masks)
            # loss_mask = ((trace_masks!=0) * (trace_feats[:,:,4]!=1)).unsqueeze(2) # for those words without labels ([0,0,1,1,1]), don't calculate the loss
            # loss = (torch.abs(outputs[:,:,:4] -  trace_feats[:,:,:4]) * loss_mask).sum() / (loss_mask.sum() * 4)
            # construct the localized optimal transport
            # D = torch.abs(outputs[:,:,:4].unsqueeze(2) - trace_feats[:,:,:4].unsqueeze(1)).mean(dim=-1)
            # T = local_OT(D).to(outputs.device)
            # loss = (torch.abs(torch.matmul(outputs[:, :, :4].transpose(1,2), T).transpose(1,2) -
            #                   trace_feats[:, :, :4]) * loss_mask).sum() / (loss_mask.sum() * 4)

            # for trace generation - classification
            trace_class_label = trace_feats[:, :, 0] * (trace_feats[:, :, 5] != 1).float() - 1
            trace_class_label = trace_class_label.view(-1).long()
            model_outputs = self.model(fc_feats, att_feats, trace_feats[:,:,1:], box_feats, labels[..., :-1], att_masks, trace_masks)
            model_outputs = F.log_softmax(model_outputs, dim=-1)
            model_outputs = model_outputs.view(-1, model_outputs.shape[2])
            loss = F.nll_loss(model_outputs, trace_class_label, ignore_index=-1)
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
