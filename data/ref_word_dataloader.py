import torch
from torch.utils.data import DataLoader
import others.util as util
import numpy as np
import random
from data.batch_data import ClarifyQuestionBatch

class RefWordsDataloader(DataLoader):
    def __init__(self, args, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, pin_memory=False,
                 drop_last=False, timeout=0, worker_init_fn=None):
        super(RefWordsDataloader, self).__init__(
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
        return self.get_batch(batch)

    def get_batch(self, batch):
        # CLS query SEP cq SEP ans ... SEP cq
        #[topic_facet_id, hist_cqs, doc_list, [pos_cq, other_cq, neg_cq]]
        topic_facet_ids = [entry[0] for entry in batch]
        topic_ids = [x.split("-")[0] for x in topic_facet_ids]
        topic_queries = [self.global_data.topic_dic[x] for x in topic_ids]
        hist_cq_ids = [entry[1] for entry in batch] # batch_size, hist_cq_count
        candi_cq_ids = [entry[3] for entry in batch]
        candi_labels = []
        if len(batch[0]) > 4:
            candi_labels = [entry[4] for entry in batch]
            candi_labels = util.pad(candi_labels, 0.)
        word_weights = []
        if len(batch[0]) > 5:
            word_weights = [entry[5] for entry in batch]
            word_weights = util.pad(word_weights, 0)
            ref_words = [entry[2] for entry in batch]
            ref_words = util.pad_3d(ref_words, self.pad_vid, dim=1)
        init_candi_scores = []
        if len(batch[0]) > 6:
            init_candi_scores = [entry[6] for entry in batch]
            init_candi_scores = util.pad(init_candi_scores, 0.)

        batch_cls = []
        candi_seg_ids, ref_doc_words, candi_cq_words = [], [], []
        for i in range(len(batch)):
            entry = batch[i]
            word_seq = [self.cls_vid] + topic_queries[i] + [self.sep_vid]
            # ref_doc_words.append([[self.cls_vid] + self.global_data.clarify_q_dic[cq] for cq in entry[2]])
            for words in ref_words[i]:
                word_seq.extend([self.cls_vid] + words + [self.sep_vid])
            batch_cls.append([idx for idx in range(1, len(word_seq)) if word_seq[idx] == self.cls_vid])

            seg_id = [0] * len(word_seq)
            # ref_doc_words.append([[self.cls_vid] + self.global_data.doc_dic[doc] for doc in entry[2]])
            per_candi_cq, per_candi_seg = [], []
            for cq in entry[3]: # candidate cq
                cq_words = self.global_data.clarify_q_dic[cq]
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
        # ref_doc_words = util.pad_3d(ref_doc_words, self.pad_vid, dim=1)
        # ref_doc_words = util.pad_3d(ref_doc_words, self.pad_vid, dim=2)
        batch = ClarifyQuestionBatch(
            topic_facet_ids, candi_cq_ids, hist_cq_ids, candi_labels, \
                candi_cq_words, candi_seg_ids, ref_doc_words=ref_doc_words, \
                    init_candi_scores=init_candi_scores, word_weights=word_weights, \
                        cls_idxs=batch_cls)
        return batch
