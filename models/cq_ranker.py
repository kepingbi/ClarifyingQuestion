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
from others.logging import logger

def build_optim(args, model, checkpoint):
    """ Build optimizer """
    saved_optimizer_state_dict = None

    if args.train_from != '' and checkpoint is not None:
        optim = checkpoint['optim']
        saved_optimizer_state_dict = optim.optimizer.state_dict()
    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method=args.decay_method,
            warmup_steps=args.warmup_steps * args.gradient_accumulation_steps,
            weight_decay=args.l2_lambda)
        #self.start_decay_steps take effect when decay_method is not noam

    optim.set_parameters(list(model.named_parameters()))

    if args.train_from != '' and checkpoint is not None:
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.device == "cuda":
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    return optim

class Bert(nn.Module):
    def __init__(self, model_path="", temp_dir="/tmp"):
        super(Bert, self).__init__()
        if os.path.exists(model_path):
            self.model = BertModel.from_pretrained(model_path)
        else:
            self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)

    def forward(self, x, segs=None, mask=None):
        encoded_layers, _ = self.model(x, token_type_ids=segs, attention_mask=mask)
        top_vec = encoded_layers[-1]
        return top_vec

class ClarifyQuestionRanker(nn.Module):
    def __init__(self, args, device):
        super(ClarifyQuestionRanker, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.pretrained_bert_path) #, args.temp_dir
        self.embedding_size = self.bert.model.config.hidden_size
        self.seg_pad_id = 0 # seg_pad_id # 0
        self.pad_vid = 0 # pad_vid # should be 0
        if self.args.model_name == "ref_transformer":
            self.transformer_encoder = TransformerRefEncoder(
                self.embedding_size, args.ff_size, \
                    args.heads, args.dropout, args.inter_layers)
        else:
        # elif self.args.model_name == "plain_transformer":
            self.wo = nn.Linear(self.embedding_size, 1, bias=True)
            if self.args.inter_embed_size > 1:
                self.wo1 = nn.Linear(self.embedding_size, self.args.inter_embed_size, bias=True)
                self.wo2 = nn.Linear(self.args.inter_embed_size, 1, bias=True)
            if self.args.model_name == "avg_transformer":
                    self.double_wo = nn.Linear(2 * self.embedding_size, 1, bias=True)
                    if self.args.inter_embed_size > 1:
                        self.double_wo1 = nn.Linear(2 * self.embedding_size, self.args.inter_embed_size, bias=True)
                        self.double_wo2 = nn.Linear(self.args.inter_embed_size, 1, bias=True)

        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)

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

    def mlp(self, hidden_emb):
        if self.args.inter_embed_size == 1:
            scores = self.wo(hidden_emb)
        else:
            scores = self.wo2(torch.relu(self.wo1(hidden_emb)))
        return scores

    def get_candi_cq_scores(self, batch_data):
        batch_size, candi_size, seq_length = batch_data.candi_cq_words.size()
        _, ref_doc_count, doc_length = batch_data.ref_doc_words.size()
        # print(ref_doc_count)
        # batch_size, candi_size, seq_length
        # batch_size, ref_doc_count, doc_length

        candi_cq_words = batch_data.candi_cq_words.view(-1, seq_length)
        candi_seg_ids = batch_data.candi_seg_ids.view(-1, seq_length)
        ref_doc_words = batch_data.ref_doc_words.view(-1, doc_length)
        doc_token_masks = ref_doc_words.ne(self.pad_vid)
        cq_words_masks = candi_cq_words.ne(self.pad_vid)
    
        query_vecs = self.bert(candi_cq_words, candi_seg_ids, cq_words_masks)
        # query_vecs is batch_size * candi_size, cq_words_len, embedding_size
        if self.args.model_name == "ref_transformer":
            ref_doc_vecs = self.bert(ref_doc_words, mask=doc_token_masks)
            # batch_size * ref_doc_count, doc_len, embedding_size
            ref_doc_vecs = ref_doc_vecs.view(batch_size, ref_doc_count, doc_length, -1)
            ref_doc_vecs = ref_doc_vecs.unsqueeze(1).expand(-1, candi_size, -1, -1, -1)
            ref_doc_vecs = ref_doc_vecs.contiguous().view(batch_size * candi_size, ref_doc_count, doc_length, -1)
            doc_token_masks = doc_token_masks.view(batch_size, ref_doc_count, -1)
            doc_token_masks = doc_token_masks.unsqueeze(1).expand(-1, candi_size, -1, -1)
            doc_token_masks = doc_token_masks.contiguous().view(batch_size * candi_size, ref_doc_count, -1)
            
            scores = self.transformer_encoder(query_vecs, ref_doc_vecs, cq_words_masks, doc_token_masks)
        elif self.args.model_name == "plain_transformer":
            cls_vecs = query_vecs.view(batch_size, candi_size, seq_length, -1)[:,:,0,:]
            scores = self.mlp(cls_vecs)
            # if self.args.inter_embed_size == 1:
            #     scores = self.wo(cls_vecs)
            # else:
            #     scores = self.wo2(torch.relu(self.wo1(cls_vecs)))
        elif self.args.model_name == "avg_transformer":
            first_vecs = query_vecs.view(batch_size, candi_size, seq_length, -1)[:,:,0,:]
            hist_cq_count = batch_data.cls_idxs.size(-1)
            if hist_cq_count > 0:
                cls_idxs = batch_data.cls_idxs.unsqueeze(1).expand(-1, candi_size, -1)
                cls_idxs = cls_idxs.contiguous().view(-1, hist_cq_count)
                hist_cq_vecs = query_vecs[torch.arange(batch_size*candi_size).unsqueeze(1), cls_idxs]
                # batch_size * candi_size, hist_cq_count, embedding_size
                cls_masks = cls_idxs.ne(0).unsqueeze(-1).float()
                hist_cq_vecs = hist_cq_vecs * cls_masks
                avg_hist_cq_vecs = torch.sum(hist_cq_vecs * cls_masks, dim=1) / cls_masks.sum(dim=1)
                # batch_size * candi_size, embedding_size
                avg_hist_cq_vecs = avg_hist_cq_vecs.view(batch_size, candi_size, -1)
                final_hidden = torch.cat([first_vecs, avg_hist_cq_vecs], dim=-1)
                if self.args.inter_embed_size == 1:
                    scores = self.double_wo(final_hidden)
                else:
                    scores = self.double_wo2(torch.relu(self.double_wo1(final_hidden)))
            else:
                scores = self.mlp(first_vecs)
                # if self.args.inter_embed_size == 1:
                #     scores = self.wo(first_vecs)
                # else:
                #     scores = self.wo2(torch.relu(self.wo1(first_vecs)))

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
            logger.info(" CQRanker initialization started.")
        if hasattr(self, "transformer_encoder"):
            self.transformer_encoder.initialize_parameters(logger)
        # if hasattr(self, "wo"):
        #     nn.init.xavier_normal_(self.wo.weight)
        #     nn.init.constant_(self.wo.bias, 0)
        # if self.args.inter_embed_size > 1:
        #     nn.init.xavier_normal_(self.wo1.weight)
        #     nn.init.constant_(self.wo1.bias, 0)
        #     nn.init.xavier_normal_(self.wo2.weight)
        #     nn.init.constant_(self.wo2.bias, 0)
        for name, p in self.named_parameters():
            if "wo" in name:
                if p.dim() > 1:
                    if logger:
                        logger.info(" {} ({}): Xavier normal init.".format(
                            name, ",".join([str(x) for x in p.size()])))
                    nn.init.xavier_normal_(p)
                else:
                    nn.init.constant_(p, 0)
                    if logger:
                        logger.info(" {} ({}): constant (0) init.".format(
                            name, ",".join([str(x) for x in p.size()])))
        if logger:
            logger.info(" CQRanker initialization finished.")

