import torch
from torch.utils.data import DataLoader
import others.util as util
import numpy as np
import random
from data.batch_data import ClarifyQuestionBatch

class ClarifyQuestionDataloader(DataLoader):
    def __init__(self, args, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, pin_memory=False,
                 drop_last=False, timeout=0, worker_init_fn=None):
        super(ClarifyQuestionDataloader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
            batch_sampler=batch_sampler, num_workers=num_workers,
            pin_memory=pin_memory, drop_last=drop_last, timeout=timeout,
            worker_init_fn=worker_init_fn, collate_fn=self._collate_fn)
        self.args = args
        self.global_data = self.dataset.global_data
        self.prod_data = self.dataset.prod_data
        self.sep_vid = self.global_data.sep_vid
        self.cls_vid = self.global_data.cls_vid
        self.pad_vid = self.global_data.pad_vid
        self.seg_pad_id = 0

    def _collate_fn(self, batch):
        # if self.prod_data.set_name == 'train':
        if self.args.model_name == "QPP":
            return self.get_batch_QPP(batch)
        else:
            return self.get_batch(batch)

    def get_batch(self, batch):
        # CLS query SEP cq SEP ans ... SEP cq
        #[topic_facet_id, hist_cqs, doc_list, [pos_cq, other_cq, neg_cq]]
        topic_facet_ids = [entry[0] for entry in batch]
        topic_ids = [x.split("-")[0] for x in topic_facet_ids]
        true_topic_facet_ids = ["%s-%s" % (x.split("-")[0], x.split("-")[1]) for x in topic_facet_ids]
        topic_queries = [self.global_data.topic_dic[x] for x in topic_ids]
        hist_cq_ids = [entry[1] for entry in batch] # batch_size, hist_cq_count
        candi_cq_ids = [entry[3] for entry in batch]
        candi_labels = []
        batch_cls = []
        if len(batch[0]) > 4:
            candi_labels = [entry[4] for entry in batch]

        candi_seg_ids, ref_doc_words, candi_cq_words = [], [], []
        for i in range(len(batch)):
            entry = batch[i]
            word_seq = [self.cls_vid] + topic_queries[i]
            for cq in entry[1]: #hist_cqs
                word_seq.extend([self.sep_vid, self.cls_vid] + self.global_data.clarify_q_dic[cq])
                if cq in self.global_data.answer_dic[true_topic_facet_ids[i]]:
                    # if this q is for other topic, it will not have corresponding answer. 
                    word_seq.extend([self.sep_vid] + self.global_data.answer_dic[true_topic_facet_ids[i]][cq])

            batch_cls.append([idx for idx in range(1, len(word_seq)) if word_seq[idx] == self.cls_vid])
            seg_id = [0] * len(word_seq)
            '''
            cur_ref_doc_words = []
            for doc in entry[2]:
                orig_topic_id = "%s-X" % topic_ids[i]
                if doc in self.global_data.doc_psg_dic[orig_topic_id]:
                    doc_words = self.global_data.doc_psg_dic[orig_topic_id][doc]
                else:
                    doc_words = self.global_data.doc_psg_dic["None"][doc]
                cur_ref_doc_words.append([self.cls_vid] + doc_words)
            # doc_topk, word_count
            ref_doc_words.append(cur_ref_doc_words)
            '''
            # ref_doc_words.append([[self.cls_vid] + self.global_data.doc_dic[doc] for doc in entry[2]])
            if len(entry[2]) > 0:
                ref_doc_words.append([[self.cls_vid] + self.global_data.clarify_q_dic[cq] for cq in entry[2]])
            else:
                # print(entry[0])
                topic = entry[0].split('-')[0]
                cq_list = self.global_data.cq_cq_rank_dic["%s-X" % topic]
                # print(cq_list[:10])
                ref_doc_words.append([[self.pad_vid]])
            per_candi_cq, per_candi_seg = [], []
            for cq in entry[3]: # candidate cq
                cq_words = [self.sep_vid] + self.global_data.clarify_q_dic[cq]
                per_candi_cq.append(word_seq + cq_words)
                per_candi_seg.append(seg_id + [1] * len(cq_words))
            candi_cq_words.append(per_candi_cq)
            candi_seg_ids.append(per_candi_seg)
        # batch_size, candi_size, seq_length
        # batch_size, ref_doc_count, doc_length
        candi_seg_ids = util.pad_3d(candi_seg_ids, self.seg_pad_id, dim=1)
        candi_seg_ids = util.pad_3d(candi_seg_ids, self.seg_pad_id, dim=2)
        candi_cq_words = util.pad_3d(candi_cq_words, self.pad_vid, dim=1)
        candi_cq_words = util.pad_3d(candi_cq_words, self.pad_vid, dim=2)
        ref_doc_words = util.pad_3d(ref_doc_words, self.pad_vid, dim=1)
        ref_doc_words = util.pad_3d(ref_doc_words, self.pad_vid, dim=2)
        batch_cls = util.pad(batch_cls, 0) # batch_size, hist_cq_count
        batch = ClarifyQuestionBatch(
            topic_facet_ids, candi_cq_ids, hist_cq_ids, candi_labels, \
                candi_cq_words, candi_seg_ids, ref_doc_words=ref_doc_words,\
                    cls_idxs=batch_cls)
        return batch

    def get_batch_QPP(self, batch):
        # CLS query SEP cq SEP ans ... SEP cq
        #[topic_facet_id, hist_cqs, doc_list, [pos_cq, other_cq, neg_cq]]
        topic_facet_ids = [entry[0] for entry in batch]
        topic_ids = [x.split("-")[0] for x in topic_facet_ids]
        true_topic_facet_ids = ["%s-%s" % (x.split("-")[0], x.split("-")[1]) for x in topic_facet_ids]
        topic_queries = [self.global_data.topic_dic[x] for x in topic_ids]
        stemmed_topic_queries = [self.global_data.stemmed_topic_dic[x] for x in topic_ids]
        hist_cq_ids = [entry[1] for entry in batch] # batch_size, hist_cq_count
        candi_cq_ids = [entry[3] for entry in batch]
        candi_labels = []
        batch_cls, batch_candi_cq_cls = [], []
        if len(batch[0]) > 4:
            candi_labels = [entry[4] for entry in batch]

        candi_seg_ids, ref_doc_words, candi_cq_words = [], [], []
        candi_retrieval_scores, candi_scores_std = [], []
        for i in range(len(batch)):
            entry = batch[i]
            hist_ql_tokens = []
            word_seq = [self.cls_vid] + topic_queries[i]
            for cq in entry[1]: #hist_cqs
                word_seq.extend([self.sep_vid, self.cls_vid] + self.global_data.clarify_q_dic[cq])
                hist_ql_tokens.extend(self.global_data.stemmed_cq_dic[cq])
                if cq in self.global_data.answer_dic[true_topic_facet_ids[i]]:
                    # if this q is for other topic, it will not have corresponding answer. 
                    word_seq.extend([self.sep_vid] + self.global_data.answer_dic[true_topic_facet_ids[i]][cq])
                    # answers can only be no. 
                    # hist_ql_tokens.extend(self.global_data.stemmed_answer_dic[true_topic_facet_ids[i]][cq])

            batch_cls.append([idx for idx in range(1, len(word_seq)) if word_seq[idx] == self.cls_vid])
            seg_id = [0] * len(word_seq)
            # ref_doc_words.append([[self.cls_vid] + self.global_data.doc_dic[doc] for doc in entry[2]])
            per_candi_cq, per_candi_seg = [], []
            per_candi_rscores, per_candi_score_std = [], []
            batch_candi_cq_cls.append(len(word_seq)+1) # the cls token before candidate cq
            for cq in entry[3]: # candidate cq
                cq_words = [self.sep_vid, self.cls_vid] + self.global_data.clarify_q_dic[cq]
                per_candi_cq.append(word_seq + cq_words)
                per_candi_seg.append(seg_id + [1] * len(cq_words))
                cur_ql_token = hist_ql_tokens + self.global_data.stemmed_cq_dic[cq]
                self.global_data.ql.update_query_lang_model(
                    query=stemmed_topic_queries[i],
                    question=cur_ql_token,
                    answer=[], alpha=self.args.ql_alpha)

                ranklists = self.global_data.ql.get_result_list(
                    topic=topic_ids[i], topk=self.args.ql_doc_topk) #[(doc_id, score)]
                retrieval_scores = [x for _,x in ranklists]
                scores_std = [np.std(retrieval_scores[:x+1]) for x in range(len(retrieval_scores))]
                per_candi_rscores.append(retrieval_scores)
                per_candi_score_std.append(scores_std)
            candi_retrieval_scores.append(per_candi_rscores)
            candi_scores_std.append(per_candi_score_std)
            candi_cq_words.append(per_candi_cq)
            candi_seg_ids.append(per_candi_seg)
        # batch_size, candi_size, seq_length
        # batch_size, ref_doc_count, doc_length
        candi_seg_ids = util.pad_3d(candi_seg_ids, self.seg_pad_id, dim=1)
        candi_seg_ids = util.pad_3d(candi_seg_ids, self.seg_pad_id, dim=2)
        candi_cq_words = util.pad_3d(candi_cq_words, self.pad_vid, dim=1)
        candi_cq_words = util.pad_3d(candi_cq_words, self.pad_vid, dim=2)
        batch_cls = util.pad(batch_cls, 0) # batch_size, hist_cq_count
        candi_retrieval_scores = util.pad_3d(candi_retrieval_scores, 0, dim=1)
        candi_scores_std = util.pad_3d(candi_scores_std, 0, dim=1)
        batch = ClarifyQuestionBatch(
            topic_facet_ids, candi_cq_ids, hist_cq_ids, candi_labels, 
            candi_cq_words, candi_seg_ids, ref_doc_words=ref_doc_words,
            cls_idxs=batch_cls, candi_cls_idxs=batch_candi_cq_cls,
            candi_retrieval_scores=candi_retrieval_scores,
            candi_scores_std=candi_scores_std)
        return batch
