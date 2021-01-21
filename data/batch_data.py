import torch

class ClarifyQuestionBatch(object):
    def __init__(self, topic_facet_ids, candi_cq_ids, hist_cq_ids, candi_labels, \
        candi_cq_words, candi_seg_ids, ref_doc_words=[], \
            candi_hist_words=[], candi_hist_segs=[], \
                init_candi_scores=[], word_weights=[], \
                    cls_idxs=[], to_tensor=True): #"cpu" or "cuda"
        self.topic_facet_ids = topic_facet_ids
        self.candi_cq_ids = candi_cq_ids
        self.candi_cq_words = candi_cq_words
        self.candi_seg_ids = candi_seg_ids
        self.hist_cq_ids = hist_cq_ids
        self.candi_labels = candi_labels
        self.ref_doc_words = ref_doc_words
        self.candi_hist_words = candi_hist_words
        self.candi_hist_segs = candi_hist_segs
        self.init_candi_scores = init_candi_scores
        self.word_weights = word_weights
        self.cls_idxs = cls_idxs

        if to_tensor:
            self.to_tensor()

    def to_tensor(self):
        self.candi_cq_words = torch.tensor(self.candi_cq_words)
        self.candi_seg_ids = torch.tensor(self.candi_seg_ids)
        self.ref_doc_words = torch.tensor(self.ref_doc_words)
        self.candi_labels = torch.tensor(self.candi_labels)
        self.candi_hist_segs = torch.tensor(self.candi_hist_segs)
        self.candi_hist_words = torch.tensor(self.candi_hist_words)
        self.init_candi_scores = torch.tensor(self.init_candi_scores)
        self.word_weights = torch.tensor(self.word_weights)
        self.cls_idxs = torch.tensor(self.cls_idxs)

    def to(self, device):
        if device == "cpu":
            return self
        else:
            candi_cq_words = self.candi_cq_words.to(device)
            candi_seg_ids = self.candi_seg_ids.to(device)
            ref_doc_words = self.ref_doc_words.to(device)
            candi_labels = self.candi_labels.to(device)
            candi_hist_segs = self.candi_hist_segs.to(device)
            candi_hist_words = self.candi_hist_words.to(device)
            init_candi_scores = self.init_candi_scores.to(device)
            word_weights = self.word_weights.to(device)
            cls_idxs = self.cls_idxs.to(device)

            return self.__class__(self.topic_facet_ids, self.candi_cq_ids, self.hist_cq_ids, candi_labels, \
                candi_cq_words, candi_seg_ids, ref_doc_words, candi_hist_words, candi_hist_segs, \
                    init_candi_scores, word_weights, cls_idxs, to_tensor=False)
