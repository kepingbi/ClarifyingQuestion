""" transformer based on reviews
    Q+r_{u1}+r_{u2} <> r_1, r_2 (of a target i)
"""
"""
review_encoder
query_encoder
transformer
"""
import os
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertConfig
from models.text_encoder import AVGEncoder, FSEncoder
from models.transformer import TransformerEncoder
from models.inter_transformer import TransformerRefEncoder
from models.optimizers import Optimizer
from models.cq_ranker import ClarifyQuestionRanker
from others.logging import logger

class RefWordsRanker(nn.Module):
    def __init__(self, args, device):
        super(RefWordsRanker, self).__init__()
        self.args = args
        self.device = device
        # self.bert = Bert(args.pretrained_bert_path) #, args.temp_dir
        self.cq_bert_ranker = ClarifyQuestionRanker(args, device)
        self.embedding_size = self.cq_bert_ranker.embedding_size
        self.seg_pad_id = 0 # seg_pad_id # 0
        self.pad_vid = 0 # pad_vid # should be 0
        
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)
        self.linear1 = nn.Linear(self.embedding_size, self.args.hidden_size, bias=True)
        if self.args.hidden_size > 1:
            self.linear2 = nn.Linear(self.args.hidden_size, 1, bias=True)

        #for each q,u,i
        #Q, previous purchases of u, current available reviews for i, padding value
        #self.logsoftmax = torch.nn.LogSoftmax(dim = -1)
        #self.bce_logits_loss = torch.nn.BCEWithLogitsLoss(reduction='none')#by default it's mean

        self.initialize_parameters(logger) #logger
        self.to(device) #change model in place


    def load_cp(self, pt, strict=True):
        self.load_state_dict(pt['model'], strict=strict)

    def test(self, batch_data):
        #topic_facet_id, hist_cqs, cur_candi_scores
        candi_scores, candi_cq_mask = self.get_candi_cq_scores(batch_data)
        return candi_scores, candi_cq_mask

    def get_candi_cq_scores(self, batch_data):
        batch_size, candi_size, seq_length = batch_data.candi_cq_words.size()
        candi_cq_words = batch_data.candi_cq_words.view(-1, seq_length)
        candi_seg_ids = batch_data.candi_seg_ids.view(-1, seq_length)
        cq_words_masks = candi_cq_words.ne(self.pad_vid)
        # query_vecs is batch_size * candi_size, cq_words_len, embedding_size
        # batch_data.cls_idxs is batch_size, ref_words_count
        query_vecs = self.cq_bert_ranker.bert(candi_cq_words, candi_seg_ids, cq_words_masks)
        first_vecs = query_vecs.view(batch_size, candi_size, seq_length, -1)[:,:,0,:]

        if self.args.selector == "ref":
            if self.args.hidden_size > 1:
                scores = self.linear2(torch.relu(self.linear1(first_vecs)))
            else:
                scores = self.linear1(first_vecs)
        else: # "wref"
            ref_word_count = batch_data.cls_idxs.size(-1)
            cls_idxs = batch_data.cls_idxs.unsqueeze(1).expand(-1, candi_size, -1)
            cls_idxs = cls_idxs.contiguous().view(-1, ref_word_count)
            ref_words_vecs = query_vecs[torch.arange(batch_size*candi_size).unsqueeze(1), cls_idxs]
            # batch_size * candi_size, word_count, embedding_size
            word_weights = batch_data.word_weights / batch_data.word_weights.sum(dim=-1, keepdim=True)
            word_weights = word_weights.unsqueeze(1).expand(-1, candi_size, -1)
            word_weights = word_weights.contiguous().view(-1, ref_word_count, 1)
            # batch_size * candi_size, word_count
            w_word_vecs = ref_words_vecs * word_weights
            w_word_vecs = w_word_vecs.sum(dim=1).view(batch_size, candi_size, -1)
            if self.args.hidden_size > 1:
                scores = self.linear2(torch.relu(self.linear1(w_word_vecs)))
            else:
                scores = self.linear1(w_word_vecs)
                

        # batch_size * candi_size
        scores = scores.view(batch_size, candi_size) + batch_data.init_candi_scores
        candi_cq_mask = cq_words_masks.view(batch_size, candi_size, -1)[:,:,0]
        return scores, candi_cq_mask

    def forward(self, batch_data):
        scores, candi_cq_mask = self.get_candi_cq_scores(batch_data)
        # targets = torch.FloatTensor([4,1,0]).to(self.device).unsqueeze(0).expand_as(scores)
        targets = batch_data.candi_labels # batch_size, candi_count
        loss = -self.logsoftmax(scores) * targets
        # loss = -self.logsoftmax(scores) * (targets.pow(2, a)-1)
        loss = loss * candi_cq_mask.float()
        loss = loss.sum(-1).mean()

        return loss

    def initialize_parameters(self, logger=None):
        if logger:
            logger.info(" RefWordsRanker initialization started.")
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0)
        if hasattr(self, "linear2"):
            nn.init.xavier_normal_(self.linear2.weight)
            nn.init.constant_(self.linear2.bias, 0)

        if logger:
            logger.info(" RefWordsRanker initialization finished.")

