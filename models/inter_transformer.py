import math

import torch
import torch.nn as nn

from models.neural import MultiHeadedAttention, \
    PositionwiseFeedForward, VariantMultiHeadedAttention

class TransformerInteractLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerInteractLayer, self).__init__()

        self.query_self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)

        self.ref_self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)

        self.cross_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)

        self.cls_cross_attn = VariantMultiHeadedAttention(
            heads, d_model, dropout=dropout)

        self.query_feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.ref_feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.query_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.ref_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, query_mask, inputs_mask):
        # query: batch_size, query_len, embed_size
        # inputs as key and value: batch_size, n_ref, key_len, embed_size
        # query_mask: batch_size, query_len
        # inputs_mask: batch_size, n_ref, key_len
        query_len = query.size(1)
        batch_size, n_ref, key_len, embed_size = inputs.size()
        inputs = inputs.view(-1, key_len, embed_size)
        if (iter != 0):
            query_norm = self.query_layer_norm(query)
            input_norm = self.ref_layer_norm(inputs)
        else:
            query_norm = query
            input_norm = inputs

        query_context = self.query_self_attn(query_norm, query_norm, query_norm,
                                 mask=1-query_mask.unsqueeze(1))
        # batch_size, query_len, embed_size
        input_context = self.ref_self_attn(input_norm, input_norm, input_norm,
                                 mask=1-inputs_mask.view(-1, key_len).unsqueeze(1))
        # batch_size * n_ref, key_len, embed_size
        query_context = self.query_layer_norm(self.dropout(query_context) + query)
        inputs_context = self.ref_layer_norm(self.dropout(input_context) + inputs)

        expd_query_context = query_context.unsqueeze(1).expand(-1, n_ref, -1, -1)
        expd_query_context = expd_query_context.contiguous().view(batch_size * n_ref, query_len, embed_size)
        expd_query_mask = query_mask.unsqueeze(1).expand(-1, n_ref, -1)
        expd_query_mask = expd_query_mask.contiguous().view(-1, query_len)

        expd_qmask = expd_query_mask.unsqueeze(-1).expand(-1,-1,key_len)
        expd_imask = inputs_mask.view(-1, key_len).unsqueeze(1).expand_as(expd_qmask)
        cross_mask = expd_qmask * expd_imask
        # cross_mask is batch_size * n_ref, query_len, key_len

        query_token_out = self.cross_attn(
            inputs_context, inputs_context, expd_query_context, mask=1-cross_mask)
        # batch_size * n_ref, query_len, embed_size
        ref_cls_emb = inputs_context[:, 0, :].view(batch_size, n_ref, embed_size)
        ref_cls_masks = inputs_mask[:,:,0] #batch_size, n_ref
        expd_qmask = query_mask.unsqueeze(-1).expand(-1,-1,n_ref)
        expd_imask = ref_cls_masks.unsqueeze(1).expand_as(expd_qmask)
        cls_cross_mask = expd_qmask * expd_imask
        # batch_size, query_len, n_ref

        value_embs = query_token_out.view(
            batch_size, n_ref, query_len, -1).transpose(1,2).contiguous().view(-1, n_ref, embed_size)
        query_cls_context = self.cls_cross_attn(
            ref_cls_emb, value_embs, query_context, mask=1-cls_cross_mask)
        
        query_cls_context = self.dropout(query_cls_context) + query_context
        query_out = self.query_feed_forward(query_cls_context)
        ref_out = self.ref_feed_forward(inputs_context)

        return query_out, ref_out

class TransformerRefEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(TransformerRefEncoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.transformer_inter = nn.ModuleList(
            [TransformerInteractLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, 1, bias=True)

    def encode(self, query_vecs, input_vecs, query_mask, input_mask):
        """ See :obj:`EncoderBase.forward()`"""

        #query_vecs is batch_size, query_len, embedding_size
        # inputs as key and value: batch_size, n_ref, key_len, embed_size

        query_vecs = query_vecs * query_mask[:,:,None].float()
        input_vecs = input_vecs * input_mask[:,:,:,None].float()

        for i in range(self.num_inter_layers):
            query_vecs, input_vecs = self.transformer_inter[i](
                i, query_vecs, input_vecs, query_mask, input_mask)

        query_vecs = self.layer_norm(query_vecs)
        #out_pos can be 0 or -1 # represent query or item in the item_transformer model
        return query_vecs

    def forward(self, query_vecs, input_vecs, query_mask, input_mask):
        """ See :obj:`EncoderBase.forward()`"""
        x = self.encode(query_vecs, input_vecs, query_mask, input_mask)
        out_emb = x[:,0,:]# x[:,0,:] will return size batch_size, d_model
        scores = self.wo(out_emb).squeeze(-1) #* mask.float()
        #batch_size
        return scores

    def initialize_parameters(self, logger=None):
        if logger:
            logger.info(" RefTransformer initialization started.")
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() > 1:
                if logger:
                    logger.info(" {} ({}): Xavier normal init.".format(
                        name, ",".join([str(x) for x in p.size()])))
                nn.init.xavier_normal_(p)
            elif "bias" in name:
                if logger:
                    logger.info(" {} ({}): constant (0) init.".format(
                        name, ",".join([str(x) for x in p.size()])))
                nn.init.constant_(p, 0)
            else:
                if logger:
                    logger.info(" {} ({}): random normal init.".format(
                        name, ",".join([str(x) for x in p.size()])))
                nn.init.normal_(p)
        if logger:
            logger.info(" RefTransformer initialization finished.")