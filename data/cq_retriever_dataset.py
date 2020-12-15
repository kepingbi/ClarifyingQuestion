import torch
from torch.utils.data import Dataset
import numpy as np
import random
import others.util as util

from collections import defaultdict

""" load training, validation and test data
topic + (cq, ans) -> cq+, cq-
with D at each step
"""

class ClarifyQuestionDataset(Dataset):
    def __init__(self, args, global_data, prod_data, hist_cq_dic=None, candi_cq_dic=None):
        self.args = args
        self.sep_vid = global_data.sep_vid
        self.cls_vid = global_data.cls_vid
        self.pad_vid = global_data.pad_vid
        self.global_data = global_data
        self.prod_data = prod_data
        if prod_data.set_name == "train":
            self._data = self.collect_train_samples(self.global_data, self.prod_data)
            # self._data = self._data[:1000] # for testing
        else:
            candidate_cq_dic = self.prod_data.candidate_cq_dic if candi_cq_dic is None else candi_cq_dic
            self._data = self.collect_test_samples(
                self.global_data, candidate_cq_dic, hist_cq_dic)

    def collect_test_samples(self, global_data, candidate_cq_dic, hist_cq_dic):
        # query, historical questiones+answers, cq candidates
        test_data = []
        candi_batch_size = self.args.candi_batch_size
        topk = self.args.doc_topk
        # historical q from outside information.
        for topic_facet_id in candidate_cq_dic:
            # hist_cqs = hist_cq_dic[topic_facet_id] if hist_cq_dic is not None else []
            if hist_cq_dic is None or topic_facet_id not in hist_cq_dic:
                hist_cqs = []
            else:
                hist_cqs = hist_cq_dic[topic_facet_id]
            hist_cqs = [cq for cq,score in hist_cqs]
            hist_cq_set = set(hist_cqs)
            candi_cq_list = list(candidate_cq_dic[topic_facet_id].difference(hist_cq_set))
            if len(candi_cq_list) == 0: #in case the candidate list is empty, query 115 doesn't matching any clarifying question. 
                continue
            topic, _ = topic_facet_id.split('-')
            doc_list = self.doc_diff(
                global_data.cq_doc_rank_dic, global_data.cq_top_doc_info_dic, \
                    topic, hist_cqs)
            doc_list = doc_list[:topk]
            seg_count = int((len(candi_cq_list) - 1) / candi_batch_size) + 1
            for i in range(seg_count):
                test_data.append([topic_facet_id, hist_cqs, doc_list,
                candi_cq_list[i*candi_batch_size:(i+1)*candi_batch_size]])

        return test_data


    def collect_train_samples(self, global_data, prod_data):
        # query -> question+, question-
        topk = self.args.doc_topk
        train_data = []
        for hist_len in range(self.args.max_hist_turn):
            entries = self.select_neg_samples(prod_data, hist_len)
            for topic_facet_id, hist_cqs, pos_cq, other_cq, neg_cq in entries:
                topic, _ = topic_facet_id.split('-')
                doc_list = self.doc_diff(
                    global_data.cq_doc_rank_dic, global_data.cq_top_doc_info_dic, \
                        topic, hist_cqs)
                doc_list = doc_list[:topk]
                train_data.append([topic_facet_id, hist_cqs, doc_list, [pos_cq, other_cq, neg_cq]])
        return train_data

    def doc_diff(self, cq_doc_rank_dic, cq_top_doc_info_dic, topic, hist_cqs):
        init_doc_list = cq_doc_rank_dic["%s-X" % topic]
        init_doc_rank = cq_top_doc_info_dic["%s-X" % topic]
        doc_list = [] if len(hist_cqs) > 0 else init_doc_list
        for cq in hist_cqs:
            cur_cq_doc_rank = cq_top_doc_info_dic[cq]
            for doc in init_doc_list:
                # other clarifying questions that are not for the current topic may not have doc
                if doc not in cur_cq_doc_rank or cur_cq_doc_rank[doc] > init_doc_rank[doc]:
                    doc_list.append(doc)
        if len(doc_list) == 0:
            doc_list = init_doc_list
        return doc_list

    def select_neg_samples(self, prod_data, hist_len):
        # hist_len: 0,1,2,3
        # cq:1->cq:2; cq:0->cq:2
        entries = []
        all_cq_set = set(self.global_data.clarify_q_dic.keys())
        rand_numbers = np.random.random(len(prod_data.pos_cq_dic) * 5)
        pos_cq_thre = self.args.pos_cq_thre # 0.8
        cur_no = 0
        for topic_facet_id in prod_data.pos_cq_dic:
            cur_pos_set, other_pos_set = prod_data.pos_cq_dic[topic_facet_id]
            candi_cq_set = prod_data.candidate_cq_dic[topic_facet_id]
            candi_cq_set = candi_cq_set.difference(cur_pos_set).difference(other_pos_set)
            global_candi_set = all_cq_set.difference(cur_pos_set).difference(other_pos_set)
            hist_cqs_set = set()
            hist_cqs = []
            if hist_len > 0:
                hist_cqs = random.sample(other_pos_set, hist_len)
                hist_cqs_set = set(hist_cqs)
                if len(candi_cq_set) >= hist_len:
                    candi_cqs = random.sample(candi_cq_set, hist_len)
                else:
                    candi_cqs = list(candi_cq_set)
                    candi_cqs += random.sample(global_candi_set, hist_len-len(candi_cq_set))

                for x in range(hist_len):
                    if rand_numbers[cur_no+x] > pos_cq_thre and candi_cqs[x] not in hist_cqs_set:
                        hist_cqs[x] = candi_cqs[x]
                cur_no += hist_len
                hist_cqs_set = set(hist_cqs)
            for pos_cq in cur_pos_set: # 1 cur_pos, 1 other_pos, 1 negative
                other_cq = random.sample(other_pos_set.difference(hist_cqs_set), 1)
                candi_cq_set = candi_cq_set.difference(hist_cqs_set)
                if len(candi_cq_set) == 0:
                    candi_cq_set = global_candi_set.difference(hist_cqs_set)
                neg_cq = random.sample(candi_cq_set, 1)
                entries.append([topic_facet_id, hist_cqs, pos_cq, other_cq[0], neg_cq[0]])
        return entries

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]
