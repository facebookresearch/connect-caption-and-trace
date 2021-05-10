# This file contains Transformer network
# Most of the code is copied from http://nlp.seas.harvard.edu/2018/04/03/attention.html

# The cfg name correspondance:
# N=num_layers
# d_model=input_encoding_size
# d_ff=rnn_size
# h is always 8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils

import copy
import math
import numpy as np

from .CaptionModel import CaptionModel
from .AttModel_both import sort_pack_padded_sequence, pad_unsort_packed_sequence, pack_wrapper, AttModel

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator_caption, generator_trace, d_model, dropout):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator_caption = generator_caption
        self.generator_trace = generator_trace
        # self.decode_layernorm =  nn.LayerNorm(d_model, elementwise_affine=True)
        # self.dropout = nn.Dropout(dropout)
        self.trace_layernorm_caption = nn.LayerNorm(d_model, elementwise_affine=True)
        self.trace_layernorm_trace = nn.LayerNorm(d_model, elementwise_affine=True)
        self.position_encoder = PositionalEncoding(d_model,0) # here don't use dropout inside positional embedding
        self.trace_embed = nn.Sequential(*(
                (nn.Linear(5, d_model),
                 nn.LayerNorm(d_model, elementwise_affine=True),
                 nn.ReLU(),
                 nn.Dropout(0.5)) ))
        self.trace_feat_embed = nn.Sequential(*(
                 (nn.Linear(2048, d_model),
                  nn.LayerNorm(d_model, elementwise_affine=True),
                 nn.ReLU(),
                 nn.Dropout(0.5))))
        
    def forward(self, src, tgt, src_mask, tgt_mask, trace_feat, trace_masks, task):
        "Take in and process masked src and target sequences."
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask, trace_feat, trace_masks, task), memory
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask, trace_feats, trace_masks, task):
        # if task == 'trace':
        ### get trace_feat
        # trace_grid_feats = trace_feats[:, :, 5:]
        trace_feats = trace_feats[:, :, :5]
        # trace_grid_feats = self.trace_feat_embed(trace_grid_feats)
        trace_feats = self.trace_embed(trace_feats)
        trace_feats = self.trace_layernorm_trace(self.position_encoder(trace_feats))


        ### embed the tgt and then add the trace_grid_feat: add trace_feat in the beginning
        tgt_emd = self.tgt_embed(tgt, task) #, task
        # if tgt.shape[1] > trace_feats.shape[1]:
        #     trace_feats = torch.cat([trace_feats, torch.zeros([trace_feats.shape[0], tgt_emd.shape[1]-trace_feats.shape[1],
        #                                                      trace_feats.shape[2]]).to(trace_feats.device)], 1)
        # else:
        #     trace_feats = trace_feats[:, :tgt_emd.shape[1], :]
        # tgt_emd = self.dropout(self.decode_layernorm(tgt_emd + trace_feat))
        return self.decoder(tgt_emd, trace_feats, memory, src_mask, tgt_mask, trace_masks, task)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.norm_2 = LayerNorm(layer.size)
        
    def forward(self, x, trace_feat, memory, src_mask, tgt_mask, trace_masks, task):
        for layer in self.layers:
            x = layer(x, trace_feat, memory, src_mask, tgt_mask, trace_masks, task)
        if task == 'both':
            return self.norm(x[0]), self.norm_2(x[1])
        else:
            return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, caption_trace_attn, trace_caption_attn, trace_self_attn, trace_src_attn,
                 feed_forward_caption, feed_forward_trace, both_caption_trace_attn, both_trace_caption_attn,
                 both_feed_forward_caption, both_feed_forward_trace,dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward_caption = feed_forward_caption
        self.feed_forward_trace = feed_forward_trace
        # self.sublayer = clones(SublayerConnection(size, dropout), 3)
        self.sublayer = clones(SublayerConnection(size, dropout), 8+4)
        ###
        self.caption_trace_attn = caption_trace_attn
        self.trace_caption_attn = trace_caption_attn
        self.trace_self_attn = trace_self_attn
        self.trace_src_attn = trace_src_attn
        ### both attn
        self.both_caption_trace_attn = both_caption_trace_attn
        self.both_trace_caption_attn = both_trace_caption_attn
        self.both_feed_forward_caption = both_feed_forward_caption
        self.both_feed_forward_trace = both_feed_forward_trace
        ###########

    def forward(self, x, trace_feat, memory, src_mask, tgt_mask, trace_masks, task):
        "Follow Figure 1 (right) for connections."
        m = memory
        if task == 'trace' or task == 'cycle_trace':
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
            ### add an layer for x to attend on trace feature
            trace_feat = self.sublayer[2](trace_feat,
                                          lambda trace_feat: self.trace_self_attn(trace_feat, trace_feat, trace_feat, trace_masks))
            trace_feat = self.sublayer[3](trace_feat,
                                          lambda trace_feat: self.trace_src_attn(trace_feat, m, m, src_mask))
            # trace_feat = self.sublayer[6](trace_feat, lambda trace_feat: self.trace_caption_attn(trace_feat, x, x, tgt_mask))
            ################################################
            return self.sublayer[7](trace_feat, self.feed_forward_trace)
        elif task == 'caption':
            m = memory
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
            ### add an layer for x to attend on trace feature
            # trace_mask = tgt_mask[:, -1, :].unsqueeze(1).long()
            trace_masks = trace_masks.unsqueeze(1)
            trace_feat = self.sublayer[2](trace_feat,
                                          lambda trace_feat: self.trace_self_attn(trace_feat, trace_feat, trace_feat,
                                                                                  trace_masks))
            trace_feat = self.sublayer[3](trace_feat,
                                          lambda trace_feat: self.trace_src_attn(trace_feat, m, m, src_mask))
            x = self.sublayer[4](x, lambda x: self.caption_trace_attn(x, trace_feat, trace_feat, trace_masks))
            ################################################
            return self.sublayer[5](x, self.feed_forward_caption)
        elif task == 'both':
            m = memory
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
            ### add an layer for x to attend on trace feature
            # trace_mask = tgt_mask[:, -1, :].unsqueeze(1).long()
            # trace_masks = trace_masks.unsqueeze(1)
            trace_feat = self.sublayer[2](trace_feat,
                                          lambda trace_feat: self.trace_self_attn(trace_feat, trace_feat, trace_feat,
                                                                                  trace_masks))
            trace_feat = self.sublayer[3](trace_feat,
                                          lambda trace_feat: self.trace_src_attn(trace_feat, m, m, src_mask))
            trace_masks_for_caption = torch.cat([trace_masks,
                                      trace_masks[:, -1, :].unsqueeze(1).repeat(1,tgt_mask.shape[1]-trace_masks.shape[1],1)], 1)
            tgt_mask_for_trace = tgt_mask[:, :trace_masks.shape[1], :]
            x_out = self.sublayer[8](x, lambda x: self.both_caption_trace_attn(x, trace_feat, trace_feat, trace_masks_for_caption))
            trace_feat_out = self.sublayer[9](trace_feat,
                                          lambda trace_feat: self.both_trace_caption_attn(trace_feat, x, x, tgt_mask_for_trace))
            return  self.sublayer[10](x_out, self.both_feed_forward_caption), self.sublayer[11](trace_feat_out, self.both_feed_forward_trace)



def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
        self.vocab = vocab
        # self.layernorm = nn.LayerNorm(self.d_model, elementwise_affine=True)

    def forward(self, x, task=None):
        if task != 'cycle_trace':
            return self.lut(x) * math.sqrt(self.d_model)
        else:
            # # use gumbel softmax with \tau = 1
            x = torch.nn.functional.softmax(torch.log(x) -
                                            torch.log(-torch.log(torch.rand([x.shape[2]]))).unsqueeze(0).unsqueeze(0).to(x.device),
                                            dim=-1)
            return torch.matmul(x, self.lut(torch.arange(self.vocab).to(x.device))) \
                   * math.sqrt(self.d_model)

class caption_Embeddings(nn.Module):
    def __init__(self, d_model, vocab, position_encoder):
        super(caption_Embeddings, self).__init__()
        self.position_encoder = position_encoder
        self.embed = Embeddings(d_model, vocab)

    def forward(self, x, task):
        x = self.embed(x, task)
        x = self.position_encoder(x)
        return x


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerModel(AttModel):

    def make_model(self, src_vocab, tgt_vocab, N_enc=6, N_dec=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        # position_nodropout = PositionalEncoding(d_model, 0)
        model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_enc),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(attn), c(attn), c(attn), c(attn),
                                 c(ff), c(ff), c(attn), c(attn), c(ff), c(ff), dropout), N_dec),
            lambda x:x, # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
            caption_Embeddings(d_model, tgt_vocab, c(position)), #nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)), #
            Generator(d_model, tgt_vocab), nn.Sequential(nn.Linear(d_model, 5), nn.Sigmoid()),
            d_model,dropout)
        
        # This was important from their code. 
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, opt):
        super(TransformerModel, self).__init__(opt)
        self.opt = opt
        # self.config = yaml.load(open(opt.config_file))


        self.N_enc = getattr(opt, 'N_enc', opt.num_layers)
        self.N_dec = getattr(opt, 'N_dec', opt.num_layers)
        self.d_model = getattr(opt, 'd_model', opt.input_encoding_size)
        self.d_ff = getattr(opt, 'd_ff', opt.rnn_size)
        self.h = getattr(opt, 'num_att_heads', 8)
        self.dropout = getattr(opt, 'dropout', 0.1)
        self.use_trace_feat = getattr(opt, 'use_trace_feat', 0)

        delattr(self, 'att_embed')
        self.att_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                    (nn.Linear(self.att_feat_size, self.d_model),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.d_model),) if self.use_bn==2 else ())))

        # define trace embedding and layernorm
        # self.trace_embed = nn.Linear(5, self.d_model)
        self.box_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(5),) if self.use_bn else ()) +
                (nn.Linear(5, self.d_model),
                 nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(self.d_model),) if self.use_bn == 2 else ())))
        self.trace_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(5),) if self.use_bn else ())+
                                    (nn.Linear(5, self.d_model),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.d_model),) if self.use_bn==2 else ())))
        self.trace_feat_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(5),) if self.use_bn else ()) +
                (nn.Linear(2048, self.d_model),
                 nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(self.d_model),) if self.use_bn == 2 else ())))
        self.box_layernorm1 = nn.LayerNorm(self.d_model, elementwise_affine=True)
        self.box_layernorm2 = nn.LayerNorm(self.d_model, elementwise_affine=True)
        self.trace_layernorm1 = nn.LayerNorm(self.d_model, elementwise_affine=True)
        self.trace_layernorm2 = nn.LayerNorm(self.d_model, elementwise_affine=True)
        self.trace_layernorm3 = nn.LayerNorm(self.d_model, elementwise_affine=True)
        self.trace_layernorm4 = nn.LayerNorm(self.d_model, elementwise_affine=True)
        self.att_layernorm = nn.LayerNorm(self.d_model, elementwise_affine=True)
        self.position_encoder = PositionalEncoding(self.d_model, self.dropout)

        
        delattr(self, 'embed')
        self.embed = lambda x : x
        delattr(self, 'fc_embed')
        self.fc_embed = lambda x : x
        delattr(self, 'logit')
        del self.ctx2att

        tgt_vocab = self.vocab_size + 1

        print(self.N_enc, self.N_dec, self.d_model, self.d_ff, self.h, self.dropout)
        self.model = self.make_model(0, tgt_vocab,
            N_enc=self.N_enc,
            N_dec=self.N_dec,
            d_model=self.d_model,
            d_ff=self.d_ff,
            h=self.h,
            dropout=self.dropout)

        c = copy.deepcopy
        # attn = MultiHeadedAttention(h, self.d_model, self.dropout)
        # ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        self.caption_embed = caption_Embeddings(self.d_model, tgt_vocab, c(position))

    def logit(self, x): # unsafe way
        return self.model.generator_caption.proj(x)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, trace_feats, box_feats, att_masks, trace_masks):


        att_feats, box_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, box_feats, att_masks)

        # Localized Narratives: insert trace features into att_feats, trace_masks into att_masks
        att_feats = self.att_layernorm(att_feats)  # normalize att feat
        if self.opt.use_box:
            box_feats = self.box_layernorm1(self.box_embed(box_feats))
            att_feats = self.box_layernorm2(att_feats + box_feats)



        if self.opt.use_trace:
            trace_feats_to_decoder = trace_feats
            if self.opt.use_trace_feat:
                trace_grid_feats = trace_feats[:, :, 5:]
                trace_feats = trace_feats[:, :, :5]
                trace_grid_feats = self.trace_layernorm3(self.trace_feat_embed(trace_grid_feats))
                # trace_grid_feats = self.position_encoder(trace_grid_feats)
                # trace_grid_feats = self.trace_layernorm4(trace_grid_feats)
            trace_feats = self.trace_layernorm1(self.trace_embed(trace_feats))
            if self.opt.use_trace_feat:
                trace_feats = trace_feats + trace_grid_feats
            # trace_feats_to_decoder = trace_feats

            trace_feats = self.position_encoder(trace_feats)  # add positional embedding
            trace_feats = self.trace_layernorm2(trace_feats)
        ### comment to test: trace feat not from encoder, only from decoder
        # att_feats = torch.cat([att_feats, trace_feats], 1)  # concat with trace feats
        # att_masks = torch.cat([att_masks, trace_masks.unsqueeze(1)], 2)

        ###########################

        memory = self.model.encode(att_feats, att_masks)

        return fc_feats[...,:0], att_feats[...,:0], memory, att_masks, trace_feats_to_decoder

    def _prepare_feature_forward(self, att_feats, box_feats, att_masks=None, seq=None):
        # comment for classification
        # att_feats, box_feats, att_masks = self.clip_att(att_feats, box_feats, att_masks)

        # original version by ruotian
        # att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        # my version: without pack and pad
        att_feats = self.att_embed(att_feats)


        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            # seq = seq[:,:-1]
            seq_mask = (seq.data != self.eos_idx) & (seq.data != self.pad_idx)
            seq_mask[:,0] = 1 # bos

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)

            seq_per_img = seq.shape[0] // att_feats.shape[0]
            if seq_per_img > 1:
                att_feats, att_masks = utils.repeat_tensors(seq_per_img,
                    [att_feats, att_masks]
                )
        else:
            seq_mask = None

        return att_feats, box_feats, seq, att_masks, seq_mask

    def _forward(self, fc_feats, att_feats, trace_feats, box_feats, seq, att_masks=None, trace_masks=None, task = None):
        assert task == 'trace' or task == 'caption' or task == 'both' or task == 'cycle_trace'
        if task != 'cycle_trace':
            if seq.ndim == 3:  # B * seq_per_img * seq_len
                seq = seq.reshape(-1, seq.shape[2])

        if task == 'both':
            ### get the original caption input
            tmp_seq = seq[:, :trace_masks.shape[1]]

            _, _, _, _, tmp_seq_mask = self._prepare_feature_forward(att_feats, box_feats, att_masks, tmp_seq)

            att_feats, box_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, box_feats,
                                                                                           att_masks, seq)

            ## prepare the shifted trace
            shifted_trace = torch.cat(
                [torch.zeros(trace_feats.shape[0], 1, trace_feats.shape[2]).to(trace_feats.device), trace_feats], 1)
            shifted_trace = shifted_trace[:, :-1, :]  # ignore the last segment in shifted trace
            shifted_trace_mask = tmp_seq_mask

            # Localized Narratives: insert trace features into att_feats, trace_masks into att_masks
            att_feats = self.att_layernorm(att_feats)  # normalize att feat
            if self.opt.use_box:
                box_feats = self.box_layernorm1(self.box_embed(box_feats))
                att_feats = self.box_layernorm2(att_feats + box_feats)

            # randomly mask part of trace_feats
            if self.training:
                random_mask_rate = 0
            else:
                random_mask_rate = 0
            random_mask = (torch.rand(
                [shifted_trace.shape[0], shifted_trace.shape[1]]) > random_mask_rate).float().unsqueeze(2).to(
                shifted_trace.device)
            shifted_trace[:, :, :5] = shifted_trace[:, :, :5] * random_mask + \
                                  (1 - random_mask) * torch.tensor([0., 0., 1., 1., 1.]).to(shifted_trace.device).unsqueeze(0).unsqueeze(1)


            # out = self.model(att_feats, seq, att_masks, seq_mask, trace_feats_to_decoder, trace_masks)
            (out_caption, out_trace), memory = self.model(att_feats, seq, att_masks, seq_mask, shifted_trace, shifted_trace_mask,
                                     task)  # trace generation

            # for regression, use generator which is a linear layer and sigmoid
            outputs_caption = self.model.generator_caption(out_caption)
            outputs_trace = self.model.generator_trace(out_trace)

            return outputs_caption, outputs_trace

        elif task == 'trace' or task == 'cycle_trace':
            # for classification
            trace_feats = trace_feats[:, :, :5]
            ### get the original caption input
            tmp_seq = torch.ones([trace_masks.shape[0], trace_masks.shape[1]]).to(trace_masks.device) # seq[:, :trace_masks.shape[1]]
            seq = seq[:, 1:trace_masks.shape[1]+1] # crop the seq to real length
            seq_mask = trace_masks.unsqueeze(1)

            att_feats, box_feats, tmp_seq, att_masks, tmp_seq_mask = self._prepare_feature_forward(att_feats, box_feats, att_masks, tmp_seq)

            ## prepare the shifted trace
            shifted_trace = torch.cat([torch.zeros(trace_feats.shape[0], 1, trace_feats.shape[2]).to(trace_feats.device), trace_feats], 1)
            shifted_trace = shifted_trace[:, :-1, :] # ignore the last segment in shifted trace
            shifted_trace_mask = tmp_seq_mask

            # Localized Narratives: insert trace features into att_feats, trace_masks into att_masks
            att_feats = self.att_layernorm(att_feats)  # normalize att feat
            if self.opt.use_box:
                box_feats = self.box_layernorm1(self.box_embed(box_feats))
                att_feats = self.box_layernorm2(att_feats + box_feats)

            # randomly mask part of trace_feats
            if self.training:
                random_mask_rate = 0
            else:
                random_mask_rate = 0
            random_mask = (torch.rand(
                [shifted_trace.shape[0], shifted_trace.shape[1]]) > random_mask_rate).float().unsqueeze(2).to(
                shifted_trace.device)
            # if torch.rand(1) > 0.5:  # half [0,0,1,1,1], half random
            shifted_trace[:, :, :5] = shifted_trace[:, :, :5] * random_mask + \
                                      (1 - random_mask) * torch.tensor([0., 0., 1., 1., 1.]).to(
                shifted_trace.device).unsqueeze(0).unsqueeze(1)
            # else:
            #     tmp_1 = torch.rand([shifted_trace.shape[0], shifted_trace.shape[1], 2]).sort(dim=2)[0]
            #     tmp_2 = torch.rand([shifted_trace.shape[0], shifted_trace.shape[1], 2]).sort(dim=2)[0]
            #     tmp = torch.stack([tmp_1[:, :, 0], tmp_2[:, :, 0], tmp_1[:, :, 1], tmp_2[:, :, 1],
            #                        (tmp_1[:, :, 1] - tmp_1[:, :, 0]) * (tmp_2[:, :, 1] - tmp_2[:, :, 0])], 2)
            #     shifted_trace[:, :, :5] = shifted_trace[:, :, :5] * random_mask + \
            #                               (1 - random_mask) * tmp.to(shifted_trace.device)


            # concat the caption into visual features
            seq_emd = self.caption_embed(seq, task)
            att_feats = torch.cat([att_feats, seq_emd], 1)
            att_masks = torch.cat([att_masks, seq_mask], 2)
            # att_masks = torch.ones([att_feats.shape[0], 1, att_feats.shape[1]]).to(att_feats.device)

            # out = self.model(att_feats, seq, att_masks, seq_mask, trace_feats_to_decoder, trace_masks)
            out, memory = self.model(att_feats, seq, att_masks, seq_mask, shifted_trace, shifted_trace_mask, task) # trace generation

            # for regression, use generator which is a linear layer and sigmoid
            outputs = self.model.generator_trace(out)

            # for classification, use (masked) dot product to provide logits
            # out = out / torch.norm(out, dim=2).unsqueeze(2)
            # memory = memory / torch.norm(memory, dim=2).unsqueeze(2)
            # outputs = torch.matmul(out, memory.transpose(1,2))
            # memory_mask = att_masks
            # outputs = outputs.masked_fill(memory_mask == 0, float('-inf'))
            #
            # outputs = F.softmax(outputs, dim=-1)
            # outputs = (outputs.unsqueeze(3) * box_feats.unsqueeze(1)).sum(dim=2)
            # print('transformer_out',outputs.argmax(dim=-1)[0])
            return outputs
        elif task == 'caption':
            att_feats, box_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, box_feats,
                                                                                           att_masks, seq)

            # Localized Narratives: insert trace features into att_feats, trace_masks into att_masks
            att_feats = self.att_layernorm(att_feats)  # normalize att feat
            if self.opt.use_box:
                box_feats = self.box_layernorm1(self.box_embed(box_feats))
                att_feats = self.box_layernorm2(att_feats + box_feats)

            if self.opt.use_trace:
                trace_feats_to_decoder = trace_feats

            out, _ = self.model(att_feats, seq, att_masks, seq_mask, trace_feats_to_decoder, trace_masks, task)

            outputs = self.model.generator_caption(out)
            return outputs
        # return torch.cat([_.unsqueeze(1) for _ in outputs], 1)

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask, trace_feats_to_decoder, trace_masks, task):
        """
        state = [ys.unsqueeze(0)]
        """
        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)

        if task == 'caption':
            out = self.model.decode(memory, mask,
                                   ys,
                                   subsequent_mask(ys.size(1)).to(memory.device),
                                    trace_feats_to_decoder, trace_masks, 'caption')
            return out[:, -1], [ys.unsqueeze(0)]

        elif task == 'both':
            out_caption, out_trace = self.model.decode(memory, mask,
                                    ys,
                                    subsequent_mask(ys.size(1)).to(memory.device),
                                    trace_feats_to_decoder, subsequent_mask(ys.size(1)).to(memory.device), 'both')
            return out_caption[:, -1], [ys.unsqueeze(0)], out_trace


