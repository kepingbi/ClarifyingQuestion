import torch
from torch.utils.data import Dataset
import numpy as np
import random, os
import others.util as util

from collections import defaultdict

""" load training, validation and test data
topic + (cq, ans) -> cq+, cq-
with D at each step
"""
class TextDataset(Dataset):
    def __init__(self, args, qrel_file, cq_dic, mode="train"):
        self.args = args
        negk = self.args.neg_per_pos
        self.qrel_dic = self.read_qrels(qrel_file)
        self.all_cqs = list(cq_dic.keys())
        # id, tokens
        self._data = []
        if mode == "train":
            for topic_id in self.qrel_dic:
                for pos_cq_id in self.qrel_dic[topic_id]:
                    neg_cq_ids = random.sample(self.all_cqs, negk)
                    self._data.append([topic_id, [pos_cq_id, neg_cq_ids]])
        else:
            candi_batch_size = args.candi_batch_size
            seg_count = int((len(self.all_cqs) - 1) / candi_batch_size) + 1
            for topic_id in self.qrel_dic:
                for i in range(seg_count):
                    self._data.append([topic_id,
                                      self.all_cqs[i*candi_batch_size:(i+1)*candi_batch_size]])

    def read_qrels(self, qrel_file):
        # read positive document set corresponding to (query_id, facet_id)
        # 1-2 Q0 1-11 1
        qrel_dict = defaultdict(set)
        with open(qrel_file, 'r') as fin:
            for line in fin:
                segs = line.strip("\r\n").split(' ')
                qid, facet_id = map(int, segs[0].split('-'))
                cq_id = segs[2]
                label = int(segs[3])
                if label > 0:
                    qrel_dict[qid].add(cq_id)
        return qrel_dict

class InitRetrievalDataset(Dataset):
    def __init__(self, args, global_data, prod_data, hist_cq_dic=None, candi_cq_dic=None):
        self.args = args
        self.sep_vid = global_data.sep_vid
        self.cls_vid = global_data.cls_vid
        self.pad_vid = global_data.pad_vid
        self.global_data = global_data
        self.prod_data = prod_data
        self.cq_topk = 10
        if prod_data.set_name == "train":
            self._data = self.collect_train_samples(self.global_data, self.prod_data)
            # self._data = self._data[:50] # for testing
        else:
            # candidate_cq_dic = self.prod_data.candidate_cq_dic if candi_cq_dic is None else candi_cq_dic
            if self.args.init_cq:
                self._data = self.collect_cq_test_samples(
                    self.global_data, self.prod_data, hist_cq_dic)
            else:
                self._data = self.collect_test_samples(
                    self.global_data, self.prod_data, hist_cq_dic)
    @staticmethod
    def read_galago_ranklist(rank_file):
        topic_ranklist_dic = defaultdict(list)
        with open(rank_file, 'r') as frank:
            for line in frank:
                segs = line.strip('\r\n').split(' ')
                doc_id = segs[2]
                topic_ranklist_dic[segs[0]].append(doc_id)
        return topic_ranklist_dic

    def collect_cq_test_samples(self, global_data, prod_data, hist_cq_dic):
        candidate_cq_dic = None
        if os.path.exists(self.args.init_rankfile):
            candidate_cq_dic = self.read_galago_ranklist(self.args.init_rankfile)
            print(len(candidate_cq_dic))
        else:
            if self.args.rerank:
                candidate_cq_dic = prod_data.candidate_cq_dic
        query_set = set()
        query_list = []
        for topic_facet_id in prod_data.pos_cq_dic: #all the topic_facet in prod_data.
            topic, _ = topic_facet_id.split('-')
            if topic not in query_set:
                query_set.add(topic)
                query_list.append("%s-X" % topic)
        print(len(query_list))
        print(len(prod_data.pos_cq_dic))
        for cq in global_data.clarify_q_dic:
            topic = cq.split("-")[0]
            if topic in query_set:
                query_list.append(cq)
        print(len(query_list))

        # query, historical questiones+answers, cq candidates
        test_data = []
        candi_batch_size = self.args.candi_batch_size
        # historical q from outside information.
        topic_candi_dic = dict()
        if candidate_cq_dic is not None:
            for topic_facet_id in candidate_cq_dic:
                topic = topic_facet_id.split("-")[0]
                topic_candi_dic[topic] = candidate_cq_dic[topic_facet_id]

        for query in query_list:
            topic = query.split("-")[0]
            if candidate_cq_dic is not None:
                candi_cq_list = list(topic_candi_dic[topic])
            else:
                candi_cq_list = list(global_data.clarify_q_dic.keys())
            if len(candi_cq_list) == 0: #in case the candidate list is empty, query 115 doesn't matching any clarifying question. 
                continue
            ref_cq_list = [cq for cq, score in global_data.cq_cq_rank_dic["%s-X" % topic]]
            ref_cq_list = ref_cq_list[:self.cq_topk]

            seg_count = int((len(candi_cq_list) - 1) / candi_batch_size) + 1
            for i in range(seg_count):
                test_data.append([query, ref_cq_list,
                candi_cq_list[i*candi_batch_size:(i+1)*candi_batch_size]])
        print(len(test_data))
        return test_data


    def collect_test_samples(self, global_data, prod_data, hist_cq_dic):
        # query, cq candidates
        test_data = []
        candi_batch_size = self.args.candi_batch_size
        # historical q from outside information.
        for topic_facet_id in prod_data.pos_cq_dic:
            if self.args.rerank:
                if topic_facet_id not in prod_data.candidate_cq_dic:
                    continue
                candi_cq_list = list(prod_data.candidate_cq_dic[topic_facet_id])
            else:
                candi_cq_list = list(global_data.clarify_q_dic.keys())
            if len(candi_cq_list) == 0: #in case the candidate list is empty, query 115 doesn't matching any clarifying question. 
                continue
            topic, _ = topic_facet_id.split('-')
            ref_cq_list = [cq for cq, score in global_data.cq_cq_rank_dic["%s-X" % topic]]
            ref_cq_list = ref_cq_list[:self.cq_topk]

            seg_count = int((len(candi_cq_list) - 1) / candi_batch_size) + 1
            for i in range(seg_count):
                test_data.append([topic_facet_id, ref_cq_list,
                candi_cq_list[i*candi_batch_size:(i+1)*candi_batch_size]])

        return test_data


    def collect_train_samples(self, global_data, prod_data):
        # query -> question+, question-
        negk = self.args.neg_per_pos
        qrel_dic = dict()
        for topic_facet_id in prod_data.pos_cq_dic:
            topic_id, _ = topic_facet_id.split("-")
            if topic_id in qrel_dic:
                continue
            cur_pos_set, other_pos_set = prod_data.pos_cq_dic[topic_facet_id]
            qrel_dic[topic_id] = cur_pos_set.union(other_pos_set)

        all_cqs = self.global_data.clarify_q_dic.keys()
        train_data = []
        for topic_id in qrel_dic:
            for pos_cq_id in qrel_dic[topic_id]:
                neg_cq_ids = random.sample(all_cqs, negk)
                ref_cq_list = [cq for cq, score in global_data.cq_cq_rank_dic["%s-X" % topic_id]]
                ref_cq_list = ref_cq_list[:self.cq_topk]

                # only calculate this for those with label 1
                train_data.append(["%s-X" % topic_id, ref_cq_list, \
                     [pos_cq_id] + neg_cq_ids, [1.] + [0.] * negk])
        return train_data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]