import torch
from torch.utils.data import Dataset
import numpy as np
import random
import others.util as util
import sys
from collections import defaultdict

""" load training, validation and test data
topic + (cq, ans) -> cq+, cq-
with D at each step
"""

class ClarifyQuestionDataset(Dataset):
    def __init__(self, args, global_data, prod_data, \
        hist_cq_dic=dict(), candi_cq_dic=None, expanded_candi_cq_dic=dict()):
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
            # if do_expansion:
            # # if self.args.fix_conv_turns == 1 and len(hist_cq_dic) == 0:
            #     # the first iteration 
            #     self.collect_init_cq_test_samples(
            #         prod_data, candidate_cq_dic, hist_cq_dic, expanded_candi_cq_dic)
            #     candidate_cq_dic = expanded_candi_cq_dic
            self._data = self.collect_test_samples(
                self.global_data, candidate_cq_dic, hist_cq_dic)
    @staticmethod
    def collect_init_cq_test_samples(prod_data, candidate_cq_dic, hist_cq_dic, expanded_candi_cq_dic):
        # enumerate all the possible clarifying questions (with label 1) for the first iteration.
        # historical q from outside information.
        for topic_facet_id in candidate_cq_dic:
            if len(candidate_cq_dic[topic_facet_id]) == 0:
                continue
            cur_pos_set, other_pos_set = prod_data.pos_cq_dic[topic_facet_id]
            # updated_tfid = "%s-X" % topic_facet_id
            # hist_cq_dic[updated_tfid] = []
            # expanded_candi_cq_dic[updated_tfid] = candidate_cq_dic[topic_facet_id]
            # prod_data.pos_cq_dic[updated_tfid] = (cur_pos_set, other_pos_set)
            for cq in other_pos_set:
                updated_tfid = "%s-%s" % (topic_facet_id, cq.split("-")[1])
                prod_data.pos_cq_dic[updated_tfid] = (cur_pos_set, other_pos_set)
                hist_cq_dic[updated_tfid] = [(cq, 0)]
                candi_cq_list = [x for x in candidate_cq_dic[topic_facet_id] if x != cq]
                expanded_candi_cq_dic[updated_tfid] = candi_cq_list

        return expanded_candi_cq_dic, hist_cq_dic

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
            # candi_cq_list = list(set(candidate_cq_dic[topic_facet_id]).difference(hist_cq_set))
            candi_cq_list = [x for x in candidate_cq_dic[topic_facet_id] if x not in hist_cq_set]

            if len(candi_cq_list) == 0: #in case the candidate list is empty, query 115 doesn't matching any clarifying question. 
                continue
            topic = topic_facet_id.split('-')[0]
            doc_list = self.doc_diff(
                global_data.cq_doc_rank_dic, global_data.cq_top_doc_info_dic, \
                    topic, hist_cqs)
            doc_list = doc_list[:topk]
            cq_list = [cq for cq,score in global_data.cq_cq_rank_dic["%s-X" % topic]]
            cq_list = [x for x in cq_list if x not in hist_cq_set]
            cq_list = cq_list[:10]

            seg_count = int((len(candi_cq_list) - 1) / candi_batch_size) + 1
            for i in range(seg_count):
                # test_data.append([topic_facet_id, hist_cqs, doc_list,
                test_data.append([topic_facet_id, hist_cqs, cq_list,
                candi_cq_list[i*candi_batch_size:(i+1)*candi_batch_size]])

        return test_data


    def collect_train_samples(self, global_data, prod_data):
        # query -> question+, question-
        topk = self.args.doc_topk
        candi_sim_wrt_tq_dic = global_data.collect_candidate_sim_wrt_tq(
            prod_data.candidate_cq_dic, global_data.cq_cq_rank_dic)

        train_data = []
        # for hist_len in range(self.args.max_hist_turn):
        for hist_len in range(self.args.min_hist_turn, self.args.max_hist_turn):
            entries = self.select_neg_samples(prod_data, hist_len)
            for topic_facet_id, hist_cqs, pos_cq, other_cq, neg_cq in entries:
            # for topic_facet_id, hist_cqs, pos_cq, neg_cq in entries:
                topic, _ = topic_facet_id.split('-')
                # doc_list = self.doc_diff(
                #     global_data.cq_doc_rank_dic, global_data.cq_top_doc_info_dic, \
                #         topic, hist_cqs)
                # doc_list = doc_list[:topk]
                cq_list = global_data.cq_cq_rank_dic["%s-X" % topic]
                cq_list = [x for x,y in cq_list if x not in set(hist_cqs)]
                cq_list = cq_list[:self.args.cq_topk]
                # if len(cq_list) == 0:
                #     print(topic_facet_id)
                # only calculate this for those with label 1
                # other_cq_sim = self.cq_similarity(global_data.cq_doc_rank_dic, hist_cqs, other_cq)
                # train_data.append([topic_facet_id, hist_cqs, doc_list, \
                train_data.append([topic_facet_id, hist_cqs, cq_list, \
                    #  [pos_cq, other_cq, neg_cq], [4., 1.-other_cq_sim, 0.]])
                     [pos_cq, other_cq, neg_cq], [4., 1., 0.]])
        return train_data

    def doc_diff(self, cq_doc_rank_dic, cq_top_doc_info_dic, topic, hist_cqs):
        init_doc_list = [doc for doc,score in cq_doc_rank_dic["%s-X" % topic]]
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

    @staticmethod
    def cq_similarity(cq_doc_rank_dic, hist_cqs, cur_cq, topk=20):
        # calculate similarity based on the portion of same documents ranked in top k documents. 
        cur_cq_ranklist = [doc for doc, score in cq_doc_rank_dic[cur_cq][:topk]]
        max_sim = 0.
        for cq in hist_cqs:
            ranklist = cq_doc_rank_dic[cq][:topk]
            overlap = set(cur_cq_ranklist).intersection(set(ranklist))
            sim = len(overlap) / topk
            max_sim = max(max_sim, sim)
        return max_sim

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
            candi_cq_set = set(prod_data.candidate_cq_dic[topic_facet_id])
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
                # entries.append([topic_facet_id, hist_cqs, pos_cq, other_cq[0]])
                # entries.append([topic_facet_id, hist_cqs, pos_cq, neg_cq[0]])
                # entries.append([topic_facet_id, hist_cqs, other_cq[0], neg_cq[0]])
        return entries

    # def collect_pointwise_train_samples(self, global_data, prod_data):
    #     # query -> question+, question-
    #     topk = self.args.doc_topk
    #     train_data = []
    #     for hist_len in range(self.args.max_hist_turn):
    #         entries = self.select_pointwise_samples(prod_data, hist_len)
    #         for topic_facet_id, hist_cqs, cq, label in entries:
    #             topic, _ = topic_facet_id.split('-')
    #             cq_list = [cq for cq, score in global_data.cq_cq_rank_dic["%s-X" % topic]]
    #             # cq_list = [x for x in cq_list if x not in set(hist_cqs)]
    #             cq_list = cq_list[:10]

    #             # only calculate this for those with label 1
    #             # other_cq_sim = self.cq_similarity(global_data.cq_doc_rank_dic, hist_cqs, other_cq)
    #             # train_data.append([topic_facet_id, hist_cqs, doc_list, \
    #             train_data.append([topic_facet_id, hist_cqs, cq_list, \
    #                 #  [pos_cq, other_cq, neg_cq], [4., 1.-other_cq_sim, 0.]])
    #                  [cq], [label]])
    #     return train_data

    # def select_pointwise_samples(self, prod_data, hist_len):
    #     # hist_len: 0,1,2,3
    #     # cq:1->cq:2; cq:0->cq:2
    #     entries = []
    #     all_cq_set = set(self.global_data.clarify_q_dic.keys())
    #     rand_numbers = np.random.random(len(prod_data.pos_cq_dic) * 5)
    #     pos_cq_thre = self.args.pos_cq_thre # 0.8
    #     cur_no = 0
    #     for topic_facet_id in prod_data.pos_cq_dic:
    #         cur_pos_set, other_pos_set = prod_data.pos_cq_dic[topic_facet_id]
    #         candi_cq_set = prod_data.candidate_cq_dic[topic_facet_id]
    #         candi_cq_set = candi_cq_set.difference(cur_pos_set).difference(other_pos_set)
    #         global_candi_set = all_cq_set.difference(cur_pos_set).difference(other_pos_set)
    #         hist_cqs_set = set()
    #         hist_cqs = []
    #         if hist_len > 0:
    #             hist_cqs = random.sample(other_pos_set, hist_len)
    #             hist_cqs_set = set(hist_cqs)
    #             if len(candi_cq_set) >= hist_len:
    #                 candi_cqs = random.sample(candi_cq_set, hist_len)
    #             else:
    #                 candi_cqs = list(candi_cq_set)
    #                 candi_cqs += random.sample(global_candi_set, hist_len-len(candi_cq_set))

    #             for x in range(hist_len):
    #                 if rand_numbers[cur_no+x] > pos_cq_thre and candi_cqs[x] not in hist_cqs_set:
    #                     hist_cqs[x] = candi_cqs[x]
    #             cur_no += hist_len
    #             hist_cqs_set = set(hist_cqs)
    #         for pos_cq in cur_pos_set: # 1 cur_pos, 1 other_pos, 1 negative
    #             entries.append([topic_facet_id, hist_cqs, pos_cq, 2])
    #         for other_cq in other_pos_set.difference(hist_cqs_set):
    #             entries.append([topic_facet_id, hist_cqs, other_cq, 1])
    #         for neg_cq in candi_cq_set.difference(hist_cqs_set):
    #             entries.append([topic_facet_id, hist_cqs, neg_cq, 0])
                
    #     return entries

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]
