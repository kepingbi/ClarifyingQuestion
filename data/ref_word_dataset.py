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

class RefWordsDataset(Dataset):
    def __init__(self, args, global_data, prod_data, hist_cq_dic=None, candi_cq_dic=None):
        self.args = args
        self.sep_vid = global_data.sep_vid
        self.cls_vid = global_data.cls_vid
        self.pad_vid = global_data.pad_vid
        self.global_data = global_data
        self.prod_data = prod_data
        self.candi_sim_wrt_tq_dic = global_data.collect_candidate_sim_wrt_tq(
            prod_data.all_candidate_cq_dic, global_data.cq_cq_rank_dic)

        if prod_data.set_name == "train":
            print(len(prod_data.all_candidate_cq_dic), len(prod_data.candidate_cq_dic))
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
            # candi_cq_set = set([x for x,score in self.prod_data.candidate_cq_dic[topic_facet_id]])
            # candi_cq_list = list(candi_cq_set.difference(hist_cq_set))
            candi_cq_list = [x for x,score in self.prod_data.candidate_cq_dic[topic_facet_id] \
                if x not in hist_cq_set]
            
            if len(candi_cq_list) == 0: #in case the candidate list is empty, query 115 doesn't matching any clarifying question. 
                continue
            cq_list = self.prod_data.all_candidate_cq_dic[topic_facet_id]
            cq_list = [cq for cq, score in cq_list[:self.args.cq_topk]]
            cq_list = [x for x in cq_list if x not in set(hist_cqs)]
            topic, _ = topic_facet_id.split('-')
            # print(topic_facet_id, hist_cqs)

            ref_word_lists, ref_word_weights = self.collect_ref_words(
                    topic, hist_cqs, cq_list, self.candi_sim_wrt_tq_dic, global_data.cq_imp_words_dic)

            # # doc_list = self.doc_diff(
            # #     global_data.cq_doc_rank_dic, global_data.cq_top_doc_info_dic, \
            # #         topic, hist_cqs)
            # # doc_list = doc_list[:topk]
            # cq_list = [cq for cq,score in global_data.cq_cq_rank_dic["%s-X" % topic]]
            # cq_list = [x for x in cq_list if x not in hist_cq_set]
            # cq_list = cq_list[:self.args.cq_topk]

            seg_count = int((len(candi_cq_list) - 1) / candi_batch_size) + 1
            for i in range(seg_count):
                seg_candi_cqs = candi_cq_list[i*candi_batch_size:(i+1)*candi_batch_size]
                candi_scores = [self.candi_sim_wrt_tq_dic[cq][topic]["%s-X" % topic] \
                    for cq in seg_candi_cqs]
                # print(seg_candi_cqs)
                # print(candi_scores)

                # test_data.append([topic_facet_id, hist_cqs, doc_list,
                test_data.append([topic_facet_id, hist_cqs, ref_word_lists,
                seg_candi_cqs, [0.] * len(seg_candi_cqs), ref_word_weights, candi_scores])

        return test_data


    def collect_train_samples(self, global_data, prod_data):
        # query -> question+, question-
        topk = self.args.doc_topk

        train_data = []
        for hist_len in range(self.args.max_hist_turn):
            entries = self.select_neg_samples(prod_data, hist_len)
            for topic_facet_id, hist_cqs, cq_tuples, labels in entries:
            # for topic_facet_id, hist_cqs, pos_cq, neg_cq in entries:
                topic, _ = topic_facet_id.split('-')
                # doc_list = self.doc_diff(
                #     global_data.cq_doc_rank_dic, global_data.cq_top_doc_info_dic, \
                #         topic, hist_cqs)
                # doc_list = doc_list[:topk]
                # cq_list = global_data.cq_cq_rank_dic["%s-X" % topic]
                cq_list = prod_data.all_candidate_cq_dic[topic_facet_id]
                cq_list = [cq for cq, score in cq_list[:self.args.cq_topk]]
                cq_list = [x for x in cq_list if x not in set(hist_cqs)]

                if len(cq_list) == 0:
                    print(topic_facet_id)
                # print(topic_facet_id, hist_cqs)
                # print(cq_tuples, labels)
                # print(candi_scores)
                ref_word_lists, ref_word_weights = self.collect_ref_words(
                    topic, hist_cqs, cq_list, self.candi_sim_wrt_tq_dic, global_data.cq_imp_words_dic)
                candi_scores = [self.candi_sim_wrt_tq_dic[cq][topic]["%s-X" % topic] \
                    for cq in cq_tuples]

                # only calculate this for those with label 1
                # other_cq_sim = self.cq_similarity(global_data.cq_doc_rank_dic, hist_cqs, other_cq)
                # train_data.append([topic_facet_id, hist_cqs, doc_list, \
                train_data.append([topic_facet_id, hist_cqs, ref_word_lists, \
                    #  [pos_cq, other_cq, neg_cq], [4., 1.-other_cq_sim, 0.]])
                     cq_tuples, labels, ref_word_weights, candi_scores])
        return train_data

    def collect_ref_words(self, topic_id, hist_cqs, init_cqs, candi_sim_wrt_tq_dic, cq_imp_words_dic):
        # candi_sim_wrt_tq_dic : {cq_id:{topic_id:{cq_id:match_score}}}
        sigmoid = torch.nn.Sigmoid()
        ref_cq_scores = []
        for ref_cq in init_cqs:
            scores = []
            if topic_id not in candi_sim_wrt_tq_dic[ref_cq]:
                print(topic_id, ref_cq)
                print(candi_sim_wrt_tq_dic[ref_cq])

            if "%s-X" % topic_id not in candi_sim_wrt_tq_dic[ref_cq][topic_id]:
                print("Error!")
                sys.exit(-1)
            sim_wrt_t = candi_sim_wrt_tq_dic[ref_cq][topic_id]["%s-X" % topic_id]
            scores.append(sim_wrt_t)
            for hist_cq in hist_cqs:
                if hist_cq not in candi_sim_wrt_tq_dic[ref_cq][topic_id]:
                    sim_wrt_hist = 0.
                else:
                    sim_wrt_hist = candi_sim_wrt_tq_dic[ref_cq][topic_id][hist_cq]
                scores.append(sim_wrt_hist)
            ref_cq_scores.append(scores)
        ref_cq_scores = torch.tensor(ref_cq_scores).to(self.args.device)
        # t_sim = torch.log(sigmoid(ref_cq_scores[:,0] * self.args.sigmoid_t))
        # hist_sim = torch.log(1 - sigmoid(ref_cq_scores[:, 1:] * self.args.sigmoid_cq))
        t_sim = sigmoid(ref_cq_scores[:,0] * self.args.sigmoid_t)
        hist_sim = sigmoid(ref_cq_scores[:, 1:] * self.args.sigmoid_cq)
        # print(t_sim, hist_sim)
        # print(tfid, tf_top_cq_dic.get(tfid, None), hist_sim)
        if hist_sim.size(-1) > 0:
            # sim = t_sim * self.args.tweight + hist_sim.mean(dim=-1) * (1 - self.args.tweight)
            max_hist_sim, _ = hist_sim.max(dim=-1)
            # sim = t_sim * (1 - max_hist_sim)
            sim = 1 - max_hist_sim
        else:
            sim = t_sim
        ref_cq_scores = sim.cpu().tolist()
        words_score_dic = defaultdict(float)
        for ref_cq, cq_score in zip(init_cqs, ref_cq_scores):
            if len(cq_imp_words_dic[ref_cq]) == 0:
                continue
            max_word_score = max([y for x,y in cq_imp_words_dic[ref_cq]])
            for word, word_score in cq_imp_words_dic[ref_cq]:
                # words_score_dic[word] += word_score * cq_score / max_word_score
                words_score_dic[word] += cq_score
        sorted_words = sorted(words_score_dic, key=words_score_dic.get, reverse=True)
        sorted_words = sorted_words[:self.args.words_topk]

        sorted_tokens = [self.global_data.tokenizer.tokenize(w) for w in sorted_words]
        token_ids = [self.global_data.tokenizer.convert_tokens_to_ids(t) for t in sorted_tokens]

        word_weights = [words_score_dic[w] for w in sorted_words]
        # print(sorted_words)
        # print(word_weights)
        return token_ids, word_weights

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

    def select_neg_samples_triples(self, prod_data, hist_len):
        # hist_len: 0,1,2,3
        # cq:1->cq:2; cq:0->cq:2
        entries = []
        # all_cq_set = set(self.global_data.clarify_q_dic.keys())
        rand_numbers = np.random.random(len(prod_data.pos_cq_dic) * 5)
        pos_cq_thre = self.args.pos_cq_thre # 0.8
        cur_no = 0
        for topic_facet_id in prod_data.pos_cq_dic:
            cur_pos_set, other_pos_set = prod_data.pos_cq_dic[topic_facet_id]
            candi_cq_set = set([x for x,score in prod_data.candidate_cq_dic[topic_facet_id]])
            # add these two line to only use the positive in the candidate set
            cur_pos_set = cur_pos_set.intersection(candi_cq_set)
            other_pos_set = other_pos_set.intersection(candi_cq_set)
            ###################
            candi_cq_set = candi_cq_set.difference(cur_pos_set).difference(other_pos_set)
            # global_candi_set = all_cq_set.difference(cur_pos_set).difference(other_pos_set)
            hist_cqs_set = set()
            hist_cqs = []
            if hist_len > 0:
                if len(other_pos_set) < hist_len or len(candi_cq_set) < hist_len:
                    continue
                hist_cqs = random.sample(other_pos_set, hist_len)
                hist_cqs_set = set(hist_cqs)
                candi_cqs = random.sample(candi_cq_set, hist_len)
                # else:
                #     candi_cqs = list(candi_cq_set)
                #     candi_cqs += random.sample(global_candi_set, hist_len-len(candi_cq_set))

                for x in range(hist_len):
                    if rand_numbers[cur_no+x] > pos_cq_thre and candi_cqs[x] not in hist_cqs_set:
                        hist_cqs[x] = candi_cqs[x]
                cur_no += hist_len
                hist_cqs_set = set(hist_cqs)
            other_pos_set = other_pos_set.difference(hist_cqs_set)
            candi_cq_set = candi_cq_set.difference(hist_cqs_set)
            for pos_cq in cur_pos_set: # 1 cur_pos, 1 other_pos, 1 negative
                if len(other_pos_set) + len(candi_cq_set) < 2:
                    continue
                if len(other_pos_set) == 0:
                    neg_cq = random.sample(candi_cq_set, 2)
                    entries.append([topic_facet_id, hist_cqs, [pos_cq, neg_cq[0], neg_cq[1]], [4.,0.,0.]])
                    continue
                if len(candi_cq_set) == 0:
                    other_cq = random.sample(other_pos_set, 2)
                    entries.append([topic_facet_id, hist_cqs, [pos_cq, other_cq[0], other_cq[1]], [4.,1.,1.]])
                    continue

                other_cq = random.sample(other_pos_set, 1)
                neg_cq = random.sample(candi_cq_set, 1)
                entries.append([topic_facet_id, hist_cqs, [pos_cq, other_cq[0], neg_cq[0]], [4.,1.,0.]])

                # if len(candi_cq_set) == 0:
                #     candi_cq_set = global_candi_set.difference(hist_cqs_set)
                # entries.append([topic_facet_id, hist_cqs, pos_cq, other_cq[0]])
                # entries.append([topic_facet_id, hist_cqs, pos_cq, neg_cq[0]])
                # entries.append([topic_facet_id, hist_cqs, other_cq[0], neg_cq[0]])
        return entries

    def select_neg_samples(self, prod_data, hist_len):
        # hist_len: 0,1,2,3
        # cq:1->cq:2; cq:0->cq:2
        entries = []
        # all_cq_set = set(self.global_data.clarify_q_dic.keys())
        # rand_numbers = np.random.random(len(prod_data.pos_cq_dic) * 5)
        # pos_cq_thre = self.args.pos_cq_thre # 0.8
        cur_no = 0
        for topic_facet_id in prod_data.pos_cq_dic:
            cur_pos_set, other_pos_set = prod_data.pos_cq_dic[topic_facet_id]
            candi_cq_set = set([x for x,score in prod_data.candidate_cq_dic[topic_facet_id]])
            # add these two line to only use the positive in the candidate set
            cur_pos_set = cur_pos_set.intersection(candi_cq_set)
            # other_pos_set = other_pos_set.intersection(candi_cq_set)
            ###################
            candi_cq_set = candi_cq_set.difference(cur_pos_set)
            # global_candi_set = all_cq_set.difference(cur_pos_set).difference(other_pos_set)
            hist_cqs_set = set()
            hist_cqs = []
            if hist_len > 0:
                if len(candi_cq_set) < hist_len:
                    continue
                hist_cqs = random.sample(candi_cq_set, hist_len)
                hist_cqs_set = set(hist_cqs)
            other_pos_set = other_pos_set.difference(hist_cqs_set)
            candi_cq_set = candi_cq_set.difference(hist_cqs_set)
            for pos_cq in cur_pos_set: # 1 cur_pos, 1 other_pos, 1 negative
                sample_count = min(len(candi_cq_set), self.args.neg_per_pos)
                if sample_count == 0:
                    continue
                sample_cqs = random.sample(candi_cq_set, sample_count)
                for other_cq in sample_cqs:
                    label = 1. if other_cq in other_pos_set else 0.
                    entries.append([topic_facet_id, hist_cqs, [pos_cq, other_cq], [4.,label]])

        return entries

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]
