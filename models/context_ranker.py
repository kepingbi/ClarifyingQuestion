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

class ContextRanker(nn.Module):
    def __init__(self, args, device):
        super(ContextRanker, self).__init__()
        self.args = args
        self.device = device
        # self.bert = Bert(args.pretrained_bert_path) #, args.temp_dir
        self.cq_bert_ranker = ClarifyQuestionRanker(args, device)
        self.embedding_size = self.cq_bert_ranker.embedding_size
        self.seg_pad_id = 0 # seg_pad_id # 0
        self.pad_vid = 0 # pad_vid # should be 0
        
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)
        self.projector = nn.Linear(self.embedding_size, 32, bias=True)
        self.linear1 = nn.Linear(32*2, 4, bias=True)
        # self.linear1 = nn.Linear(self.embedding_size*2, 4, bias=True)
        self.linear2 = nn.Linear(4, 1, bias=True)

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
        batch_size, candi_size, topic_seq_length = batch_data.candi_cq_words.size()

        if len(batch_data.candi_hist_words.size()) == 4:
            _, _, hist_len, cq_seq_length = batch_data.candi_hist_words.size()
        else:
            # print("no cq")
            return self.cq_bert_ranker.get_candi_cq_scores(batch_data)
        # _, ref_doc_count, doc_length = batch_data.ref_doc_words.size()
        # print(ref_doc_count)
        # batch_size, candi_size, seq_length
        # batch_size, ref_doc_count, doc_length
        candi_cq_words = batch_data.candi_cq_words.view(-1, topic_seq_length)
        candi_seg_ids = batch_data.candi_seg_ids.view(-1, topic_seq_length)
        cq_words_masks = candi_cq_words.ne(self.pad_vid)

        candi_hist_seq_words = batch_data.candi_hist_words.view(-1, cq_seq_length)
        candi_hist_seg_ids = batch_data.candi_hist_segs.view(-1, cq_seq_length)
        candi_hist_seq_masks = candi_hist_seq_words.ne(self.pad_vid)

        # ref_doc_words = batch_data.ref_doc_words.view(-1, doc_length)
        # doc_token_masks = ref_doc_words.ne(self.pad_vid)
        if self.args.sel_struct == "concat":
            topic_seq_vecs = self.cq_bert_ranker.bert(candi_cq_words, candi_seg_ids, cq_words_masks)
            # topic_seq_vecs is batch_size * candi_size, topic_seq_length, embedding_size
            topic_cls_vecs = topic_seq_vecs.view(
                batch_size, candi_size, topic_seq_length, -1)[:,:,0,:]
            topic_scores = self.cq_bert_ranker.wo(topic_cls_vecs)

            mapped_topic_vecs = torch.relu(self.projector(topic_cls_vecs))

            hist_seq_vecs = self.cq_bert_ranker.bert(
                candi_hist_seq_words, candi_hist_seg_ids, candi_hist_seq_masks)

            hist_seq_vecs = hist_seq_vecs.view(
                batch_size, candi_size, hist_len, cq_seq_length, -1)[:,:,:,0,:]
            # hist_cq_scores = self.cq_bert_ranker.wo(hist_seq_vecs)
            # print(topic_scores, hist_cq_scores)
            mapped_hist_vecs = torch.relu(self.projector(hist_seq_vecs))
            # pooled_vecs, _ = hist_seq_vecs.max(dim=2) # among all hist seq
            pooled_vecs, _ = mapped_hist_vecs.max(dim=2) # among all hist seq
            concat_vecs = torch.cat([mapped_topic_vecs, pooled_vecs], dim=-1)
            scores = self.linear2(torch.relu(self.linear1(concat_vecs)))
        elif self.args.sel_struct == "numeric":
            score_wrt_topics, candi_cq_mask = self.cq_bert_ranker.get_candi_cq_scores(batch_data)
            #TODO

        # batch_size * candi_size
        scores = scores.view(batch_size, candi_size)
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
            logger.info(" ContextRanker initialization started.")
        # self.cq_bert_ranker.initialize_parameters(logger)
        nn.init.xavier_normal_(self.projector.weight)
        nn.init.constant_(self.projector.bias, 0)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0)
        if logger:
            logger.info(" ContextRanker initialization finished.")

