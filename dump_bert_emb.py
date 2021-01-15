""" Output the embeddings of the CLS tokens of Queries, documents, and clarifying questions. 
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorch_pretrained_bert import BertModel, BertConfig, BertTokenizer
from models.cq_ranker import Bert
from others.logging import logger
import others.util as util
import argparse, json
from tqdm import tqdm
from collections import defaultdict
import numpy as np

class TextDataset(Dataset):
    def __init__(self, data_dic, tokenizer):
        self.tokenizer = tokenizer
        self.cls_vid = self.tokenizer.vocab['[CLS]']
        # id, tokens
        self._data = []
        for entry_id in data_dic:
            tokens = data_dic[entry_id]
            self._data.append([entry_id, [self.cls_vid] + tokens])

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]

class TextDataloader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, pin_memory=False,
                 drop_last=False, timeout=0, worker_init_fn=None):
        super(TextDataloader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
            batch_sampler=batch_sampler, num_workers=num_workers,
            pin_memory=pin_memory, drop_last=drop_last, timeout=timeout,
            worker_init_fn=worker_init_fn, collate_fn=self._collate_fn)
        self.tokenizer = self.dataset.tokenizer
        self.sep_vid = self.tokenizer.vocab['[SEP]']
        self.cls_vid = self.tokenizer.vocab['[CLS]']
        self.pad_vid = self.tokenizer.vocab['[PAD]']

    def _collate_fn(self, batch):
        entry_ids = [entry[0] for entry in batch]
        token_ids = [entry[1] for entry in batch]
        padded_tokens = util.pad(token_ids, self.pad_vid)
        return entry_ids, padded_tokens

def get_entry_scores(args, model, tokenizer, data_dic, batch_size, save_path, description):
    pad_vid = tokenizer.vocab['[PAD]']
    dataset = TextDataset(data_dic, tokenizer)
    dataloader = TextDataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    model.eval()
    with torch.no_grad():
        pbar = tqdm(dataloader)
        pbar.set_description(description)
        entry_vec_dic = dict()
        for batch_data in pbar:
            entry_ids, padded_tokens = batch_data
            padded_tokens = torch.tensor(padded_tokens).to(args.device)
            token_masks = padded_tokens.ne(pad_vid)
            token_vecs = model(padded_tokens, mask=token_masks) # batch_size, seq_length, hidden_dim
            final_vecs = token_vecs[:, 0, :]
            for eid, cls_vec in zip(entry_ids, final_vecs):
                entry_vec_dic[eid] = cls_vec.cpu()
    torch.save(entry_vec_dic, save_path)

def get_topic_tokens(data_path, tokenizer):
    topic_dic = dict()
    with open(data_path) as fin:
        fjson = json.load(fin)
        entry_ids = fjson["topic_id"].keys()
        question_ids = fjson["topic_facet_question_id"]
        topics = fjson["topic"]
        topic_id_set = set()
        for e_id in entry_ids:
            topic_id, _, qid = question_ids[e_id].split("-")
            if topic_id in topic_id_set:
                continue
            topic_id_set.add(topic_id)
            tokens = tokenizer.tokenize(topics[e_id])
            topic_dic[topic_id] = tokenizer.convert_tokens_to_ids(tokens)
    return topic_dic

def compute_topic_cq_sim(topic_vecs_path, cq_vecs_path, output_rankfile, topk=50):
    topic_vec_dic = torch.load(topic_vecs_path)
    cq_vec_dic = torch.load(cq_vecs_path)
    topic_vecs, cq_vecs = [], []
    idx2topicIDs, idx2cqIDs = [], []
    cosine = torch.nn.CosineSimilarity(dim=-1)
    for topic_id in topic_vec_dic:
        idx2topicIDs.append(topic_id)
        topic_vecs.append(topic_vec_dic[topic_id])
    for cq_id in cq_vec_dic:
        idx2cqIDs.append(cq_id)
        cq_vecs.append(cq_vec_dic[cq_id])
    topic_vecs = torch.stack(topic_vecs).cuda() # topic_count, embed_size
    cq_vecs = torch.stack(cq_vecs).cuda() # cq_count, embed_size

    topic_matrix = topic_vecs.unsqueeze(1).expand(-1, len(idx2cqIDs), -1)
    cq_matrix = cq_vecs.unsqueeze(0).expand(len(idx2topicIDs), -1, -1)
    scores = cosine(topic_matrix.contiguous(), cq_matrix.contiguous())
    sorted_scores, sorted_idxs = scores.sort(dim=-1, descending=True)
    sorted_scores = sorted_scores.cpu().numpy().tolist()
    sorted_idxs = sorted_idxs.cpu().numpy().tolist()
    ranklist_dic = dict()
    for i in range(len(idx2topicIDs)):
        ranklist = sorted_scores[i][:topk]
        indices = sorted_idxs[i][:topk]
        ranklist_dic["%s-X"%idx2topicIDs[i]] = (ranklist, indices)
    for i in range(len(idx2cqIDs)):
        cq_as_q_vec = cq_vecs[i].unsqueeze(0).expand(len(idx2cqIDs), -1).contiguous()
        # cq_count, embed_size
        cq_matrix = cq_vecs
        cq_scores = cosine(cq_as_q_vec, cq_matrix)
        scq_scores, scq_idxs = cq_scores.sort(dim=-1, descending=True)
        ranklist = scq_scores.cpu().numpy().tolist()[:topk]
        indices = scq_idxs.cpu().numpy().tolist()[:topk]
        ranklist_dic[idx2cqIDs[i]] = (ranklist, indices)

    with open(output_rankfile, "w") as fout:
        for q in sorted(ranklist_dic):
            ranklist, indices = ranklist_dic[q]
            for rank in range(topk):
                cq_id = idx2cqIDs[indices[rank]]
                score = ranklist[rank]
                line = "%s Q0 %s %d %f galago" % (q, cq_id, rank+1, score)
                fout.write("%s\n" % line)

def main(args):
    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain_path)
    topic_dic = get_topic_tokens(args.query_path, tokenizer)
    clarify_q_dic = torch.load(args.cq_token_path)
    doc_dic = torch.load(args.doc_token_path) # the first 500 words in the doc
    bert_model = Bert(args.bert_pretrain_path).to(args.device)
    save_dir = os.path.dirname(args.doc_token_path)
    get_entry_scores(args, bert_model, tokenizer, \
        topic_dic, 100, "{}/topic_vecs.pt".format(save_dir), "Topics")
    get_entry_scores(args, bert_model, tokenizer, \
        clarify_q_dic, 50, "{}/clarify_q_vecs.pt".format(save_dir), "CQ")
    get_entry_scores(args, bert_model, tokenizer, \
        doc_dic, 10, "{}/doc_vecs.pt".format(save_dir), "Docs")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_pretrain_path', default="/mnt/scratch/kbi/conv_search/qulac/data/bert_pretrain/finetuned_lm", type=str)
    parser.add_argument('--doc_token_path', default="/mnt/scratch/kbi/conv_search/qulac/data/working/candi_doc_dict.pt", type=str)
    parser.add_argument('--cq_token_path', default="/mnt/scratch/kbi/conv_search/qulac/data/working/questions_dict.pt", type=str)
    parser.add_argument('--query_path', default="/net/home/kbi/projects/conv_search/qulac/data/qulac/new_qulac.json", type=str)
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--topk', default=50, type=int)
    args = parser.parse_args()
    main(args)
    topic_vecs_path = "/mnt/scratch/kbi/conv_search/qulac/data/working/topic_vecs.pt" 
    cq_vecs_path = "/mnt/scratch/kbi/conv_search/qulac/data/working/clarify_q_vecs.pt"
    output_rankfile = "/mnt/scratch/kbi/conv_search/qulac/data/working/bert_sim_cq_init_cq.ranklist"
    compute_topic_cq_sim(topic_vecs_path, cq_vecs_path, output_rankfile, topk=args.topk)
