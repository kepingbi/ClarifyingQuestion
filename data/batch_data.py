import torch

class ClarifyQuestionBatch(object):
    def __init__(self, topic_facet_ids, candi_cq_ids, hist_cq_ids, \
        candi_cq_words, candi_seg_ids, ref_doc_words=[], to_tensor=True): #"cpu" or "cuda"
        self.topic_facet_ids = topic_facet_ids
        self.candi_cq_ids = candi_cq_ids
        self.candi_cq_words = candi_cq_words
        self.candi_seg_ids = candi_seg_ids
        self.hist_cq_ids = hist_cq_ids
        self.ref_doc_words = ref_doc_words

        if to_tensor:
            self.to_tensor()

    def to_tensor(self):
        self.candi_cq_words = torch.tensor(self.candi_cq_words)
        self.candi_seg_ids = torch.tensor(self.candi_seg_ids)
        self.ref_doc_words = torch.tensor(self.ref_doc_words)

    def to(self, device):
        if device == "cpu":
            return self
        else:
            candi_cq_words = self.candi_cq_words.to(device)
            candi_seg_ids = self.candi_seg_ids.to(device)
            ref_doc_words = self.ref_doc_words.to(device)

            return self.__class__(self.topic_facet_ids, self.candi_cq_ids, self.hist_cq_ids, \
                candi_cq_words, candi_seg_ids, ref_doc_words, to_tensor=False)
