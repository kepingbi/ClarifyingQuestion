import torch
import numpy as np

from others.logging import logger, init_logger
from collections import defaultdict
import others.util as util
from pytorch_pretrained_bert import BertTokenizer
import gzip
import os
import json

class ConvSearchData():
    def __init__(self, args, input_train_dir, set_name, global_data):
        self.args = args
        self.sep_vid = global_data.sep_vid
        self.cls_vid = global_data.cls_vid
        self.pad_vid = global_data.pad_vid
        self.neg_per_pos = args.neg_per_pos
        self.set_name = set_name
        self.global_data = global_data

        set_name = 'test' if set_name == 'valid' and not self.args.has_valid else set_name
        part_doc_file = os.path.join(input_train_dir, "%s_doc_id.txt.gz" % set_name)
        part_cq_file = os.path.join(input_train_dir, "%s_question_id.txt.gz" % set_name)
        logger.info("Load %s!" % part_doc_file)
        self.pos_doc_dic, self.candidate_doc_dic = self.read_partitition(part_doc_file)
        logger.info("Load %s!" % part_cq_file)
        self.pos_cq_dic, self.candidate_cq_dic = self.read_partitition(part_cq_file)
        for topic_facet_id in self.candidate_cq_dic:
            self.candidate_cq_dic[topic_facet_id] = self.candidate_cq_dic[topic_facet_id][:self.args.rerank_topk]
        #### load candidate from cq_rank file. ###
        # new_candi_cq_dic = dict()
        # for topic_facet_id in self.pos_cq_dic:
        #     topic = topic_facet_id.split("-")[0] + "-X"
        #     # cq_set = [x for x,y in self.global_data.cq_cq_rank_dic[topic]]
        #     # new_candi_cq_dic[topic_facet_id] = set(cq_set)
        #     new_candi_cq_dic[topic_facet_id] = self.global_data.cq_cq_rank_dic[topic]
        #     # list of (cq_id, score)
        # self.candidate_cq_dic = new_candi_cq_dic
        self.all_candidate_cq_dic = self.candidate_cq_dic
        if os.path.exists(args.init_rankfile):
            read_score = True if args.mode == "baseline" else False
            self.all_candidate_cq_dic = global_data.read_topic_cq_ranklist(args.init_rankfile, read_score)
            for topic_facet_id in self.all_candidate_cq_dic:
                self.all_candidate_cq_dic[topic_facet_id] = self.all_candidate_cq_dic[topic_facet_id][:self.args.rerank_topk]
            self.candidate_cq_dic = dict()
            for topic_facet_id in self.pos_cq_dic:
                self.candidate_cq_dic[topic_facet_id] = self.all_candidate_cq_dic[topic_facet_id]

        logger.info("ConvSearchData loaded completely!")

    def initialize_epoch(self):
        # self.neg_sample_products = np.random.randint(0, self.product_size, size = (self.set_review_size, self.neg_per_pos))
        # if self.args.model_name == "item_transformer":
        #     return
        # self.neg_sample_products = np.random.choice(self.product_size,
        #         size = (self.set_review_size, self.neg_per_pos), replace=True, p=self.product_dists)
        # construct data for a epoch, conversation history length 0, 1, 2, 3
        pass
    def reset_set_name(self, name):
        self.set_name = name

    def read_partitition(self, part_fname, is_cq=True):
        pos_unit_dic = dict()
        candidate_unit_dic = dict()
        with gzip.open(part_fname, 'rt') as fin:
            for line in fin:
                segs = line.strip('\r\n').split("\t")
                topic_facet_id = segs[0]
                cur_pos_dic = dict() # positive regarding current facet
                other_pos_dic = dict() # positive regarding the other facets of the topic
                for x in segs[1].split(";"):
                    rid, label = x.split(":")
                    label = int(label)
                    if label > 1:
                        cur_pos_dic[rid] = label
                    else:
                        other_pos_dic[rid] = label
                if segs[2] == "":
                    candi_docs = []
                else:
                    candi_docs = segs[2].split(";")
                if is_cq:
                    cur_pos_dic = set(cur_pos_dic.keys())
                    other_pos_dic = set(other_pos_dic.keys())
                    candi_docs = candi_docs

                pos_unit_dic[topic_facet_id] = (cur_pos_dic, other_pos_dic)
                candidate_unit_dic[topic_facet_id] = candi_docs
        return pos_unit_dic, candidate_unit_dic

class GlobalConvSearchData():
    ''' Read global data such as qulac json, initial ranklists, documents id-text
        Convert text in query, question, and documents to ids
    '''
    def __init__(self, args, data_path):
        if os.path.exists(args.pretrained_bert_path):
            self.tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_path)
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.sep_vid = self.tokenizer.vocab['[SEP]']
        self.cls_vid = self.tokenizer.vocab['[CLS]']
        self.pad_vid = self.tokenizer.vocab['[PAD]']

        # question_path = os.path.join(data_path, "questions.json.gz")
        # candi_doc_path = os.path.join(data_path, "candi_doc_dict.json.gz")
        question_path = os.path.join(data_path, "questions_dict.pt")
        candi_doc_path = os.path.join(data_path, "candi_doc_dict.pt")
        candi_doc_psg_path = os.path.join(data_path, "candi_doc_psg_dict.pt")
        qulac_path = os.path.join(data_path, "new_qulac.json")
        # topic_cq_doc_rankfile = os.path.join(data_path, "galago_index", "clarify_q_init_doc.mu1500.ranklist")
        topic_cq_doc_rankfile = os.path.join(data_path, "galago_index", "cq_top_doc_rerank50.ranklist")
        # topic_cq_cq_rankfile = os.path.join(data_path, "galago_index", "clarify_q_init_q.ranklist")
        if args.rerank:
            topic_cq_cq_rankfile = os.path.join(data_path, "bert.cq.ql.rerank")
        else:
            topic_cq_cq_rankfile = os.path.join(data_path, "bert.cq.rerank")
        # if os.path.exists(args.init_rankfile):
        #     topic_cq_cq_rankfile = args.init_rankfile
        cq_imp_word_file = os.path.join(data_path, "imp_cq_words.json")
        with open(cq_imp_word_file, 'r') as fin:
            self.cq_imp_words_dic = json.load(fin)

        # self.clarify_q_dic = self.read_id_content_json(question_path)
        # self.doc_dic = self.read_id_content_json(candi_doc_path)
        self.clarify_q_dic = torch.load(question_path)
        # self.doc_dic = torch.load(candi_doc_path) # the first 500 words in the doc
        self.doc_psg_dic = torch.load(candi_doc_psg_path) # the first psg and other top ranked doc
        self.topic_dic, self.answer_dic = self.read_qulac_answer(qulac_path)
        self.cq_doc_rank_dic = self.read_topic_cq_ranklist(topic_cq_doc_rankfile)
        logger.info("Loading %s" % topic_cq_cq_rankfile)
        self.cq_cq_rank_dic = self.read_topic_cq_ranklist(topic_cq_cq_rankfile)

        self.cq_top_doc_info_dic = dict()
        self.cq_top_cq_info_dic = dict()
        for qid in self.cq_doc_rank_dic:
            self.cq_top_doc_info_dic[qid] = {doc:rank+1 for rank, (doc, score) in enumerate(self.cq_doc_rank_dic[qid])}
        for qid in self.cq_cq_rank_dic:
            self.cq_top_cq_info_dic[qid] = {cq:rank+1 for rank, (cq,score) in enumerate(self.cq_cq_rank_dic[qid])}

        logger.info("GlobalConvSearchData loaded completely" )

    # def read_id_content_json(self, cq_file):
    #     count = 0
    #     with gzip.open(cq_file, 'rt') as fin:
    #         content_dict = json.load(fin)
    #         for rid in content_dict:
    #             content_tokens = self.tokenizer.tokenize(content_dict[rid])
    #             content_tokens = content_tokens[:500] # cutoff documents longer than 500
    #             # also consider using concatenated top ranked passages instead
    #             content_dict[rid] = content_tokens
    #             count += 1
    #             if count % 500 == 0:
    #                 logger.info("%d ids has been parsed!" % count)
    #     return content_dict
    @staticmethod
    def collect_candidate_sim_wrt_tq(candidate_cq_dic, cq_cq_rank_dic):
        candi_sim_wrt_tq_dic = defaultdict(dict)
        for tfid in candidate_cq_dic:
            tid, _ = tfid.split('-')
            for candi_cq in candidate_cq_dic[tfid]:
                score = 0.
                if len(candi_cq) == 2:
                    candi_cq, score = candi_cq
                candi_sim_wrt_tq_dic[candi_cq][tid] = {"%s-X" % tid: score}

        for tcqid in cq_cq_rank_dic:
            tid, cqid = tcqid.split('-')
            if cqid == "X":
                continue
            for cq_id, score in cq_cq_rank_dic[tcqid]:
                if cq_id not in candi_sim_wrt_tq_dic:
                    continue
                if tid not in candi_sim_wrt_tq_dic[cq_id]:
                    continue
                candi_sim_wrt_tq_dic[cq_id][tid][tcqid] = score
        return candi_sim_wrt_tq_dic

    def read_qulac_answer(self, qulac_file):
        topic_dic = dict() # query
        answer_dic = dict()
        with open(qulac_file) as fin:
            fjson = json.load(fin)
            entry_ids = fjson["topic_id"].keys()
            topics = fjson["topic"]
            question_ids = fjson["topic_facet_question_id"]
            answers = fjson["answer"]
            for e_id in entry_ids:
                topic_id, facet_id, qid = question_ids[e_id].split("-")
                if qid == "X":
                    continue
                if topic_id not in topic_dic:
                    tokens = self.tokenizer.tokenize(topics[e_id])
                    topic_dic[topic_id] = self.tokenizer.convert_tokens_to_ids(tokens)

                topic_facet_id = "%s-%s" % (topic_id, facet_id)
                global_qid = "%s-%s" % (topic_id, qid)
                if topic_facet_id not in answer_dic:
                    answer_dic[topic_facet_id] = dict()
                tokens = self.tokenizer.tokenize(answers[e_id])
                answer_dic[topic_facet_id][global_qid] = self.tokenizer.convert_tokens_to_ids(tokens)
        return topic_dic, answer_dic

    def read_topic_cq_ranklist(self, rank_file, read_score=True):
        topic_ranklist_dic = defaultdict(list)
        with open(rank_file, 'r') as frank:
            for line in frank:
                segs = line.strip('\r\n').split(' ')
                qid = segs[0]
                # if segs[0].endswith("X"):
                #     qid = segs[0].split('-')[0]
                doc_id = segs[2]
                score = float(segs[4])
                if read_score:
                    topic_ranklist_dic[qid].append((doc_id, score))
                else:
                    topic_ranklist_dic[qid].append(doc_id)

        return topic_ranklist_dic
