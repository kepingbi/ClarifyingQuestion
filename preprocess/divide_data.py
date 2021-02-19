""" Data partition to 5 folds. train/validation/test. An overall doc_id: text file.
    Get positive documents from all the top 10000 retrieved documents. 
"""
import os
import pickle
import gzip
import json
import argparse
import collections as coll
from initial_retrieval import read_galago_ranklist
from pytorch_pretrained_bert import BertTokenizer
import torch
import sys

def read_qrels(qrel_file):
    # read positive document set corresponding to (query_id, facet_id)
    # 1-2 Q0 clueweb09-enwp00-93-18081 1
    qrel_dict = coll.defaultdict(dict)
    with open(qrel_file, 'r') as fin:
        for line in fin:
            segs = line.strip("\r\n").split(' ')
            qid, facet_id = map(int, segs[0].split('-'))
            doc_no = segs[2]
            label = int(segs[3])
            if facet_id not in qrel_dict[qid]:
                qrel_dict[qid][facet_id] = []
            qrel_dict[qid][facet_id].append((doc_no, label))

    return qrel_dict

def output_pos_and_candidate_doc(doc_dir, output_dir, doc_set):
    os.makedirs(output_dir, exist_ok=True)
    doc_dic = dict()
    for x in os.listdir(doc_dir):
        if not x.endswith('gz'):
            continue
        print(x)
        fname = os.path.join(doc_dir, x)
        with gzip.open(fname, 'rt') as fin:
            topic_dict = json.load(fin)
            for doc_no in topic_dict['id']:
                cluebweb_id = topic_dict['id'][doc_no]
                if cluebweb_id not in doc_set:
                    continue
                if cluebweb_id in doc_dic:
                    continue
                doc_dic[cluebweb_id] = topic_dict['text'][doc_no]
    # doc_dict_file = "%s/candi_doc_dict.pkl" % (output_dir)
    # with open(doc_dict_file, 'wb') as fout:
    #     pickle.dump(doc_dic, fout)
    doc_dict_file = "%s/candi_doc_dict.json.gz" % (output_dir)
    with gzip.open(doc_dict_file, 'wt') as fout:
        json.dump(doc_dic, fout, indent=4)
        # json_doc_dic = {"id": [], "text": []}
        # for doc_no in doc_dic:
        #     json_doc_dic["id"].append(doc_no)
        #     json_doc_dic["text"].append(doc_dic[doc_no])
        # json.dump(json_doc_dic, fout)

def output_question_text(data_path, output_path):
    clarify_q_dic = dict()
    with open(data_path) as fin:
        fjson = json.load(fin)
        entry_ids = fjson["topic_id"].keys()
        question_ids = fjson["topic_facet_question_id"]
        questions = fjson["question"]
        qid_set = set()
        for e_id in entry_ids:
            topic_id, _, qid = question_ids[e_id].split("-")
            if qid == "X":
                continue
            global_qid = "%s-%s" % (topic_id, qid)
            if global_qid in clarify_q_dic:
                continue
            qid_set.add(global_qid)
            qtext = questions[e_id]
            clarify_q_dic[global_qid] = qtext

    output_file = "%s/questions.json.gz" % output_path
    with gzip.open(output_file, "wt") as fout:
        json.dump(clarify_q_dic, fout, indent=4)

def partition_to_k_folds(qrel_dict, candi_ranklist_dic, output_dir, is_q=False, k=5):
    for i in range(1, k+1):
        test_i = i % k
        dev_i = (i-1) % k
        fold_dir = os.path.join(output_dir, "fold_%d" % i)
        os.makedirs(fold_dir, exist_ok=True)
        q_or_d = "question" if is_q else "doc"
        train_file = os.path.join(fold_dir, "train_%s_id.txt.gz" % q_or_d)
        dev_file = os.path.join(fold_dir, "valid_%s_id.txt.gz" % q_or_d)
        test_file = os.path.join(fold_dir, "test_%s_id.txt.gz" % q_or_d)
        with gzip.open(train_file, "wt") as ftrain, \
            gzip.open(dev_file, "wt") as fdev, \
                gzip.open(test_file, "wt") as ftest:
            for qid in qrel_dict:
                for facet_id in qrel_dict[qid]:
                    cur_qfacet_posd_set = set([doc for doc,label in qrel_dict[qid][facet_id]])
                    if is_q:
                        cur_qfacet_posd_list = [(doc,label) for doc,label in qrel_dict[qid][facet_id]]
                    else:
                        cur_qfacet_posd_list = [(doc,label+1) for doc,label in qrel_dict[qid][facet_id]]
                        # 1 reserved for other d
                        for other_facet_id in qrel_dict[qid]:
                            if other_facet_id == facet_id:
                                continue
                            for doc_no, _ in qrel_dict[qid][other_facet_id]:
                                if doc_no in cur_qfacet_posd_set:
                                    continue
                                cur_qfacet_posd_set.add(doc_no)
                                cur_qfacet_posd_list.append((doc_no, 1))
                    cur_qfacet_candi_list = candi_ranklist_dic["%d-X" % qid]
                    output_arr = []
                    output_arr.append("%d-%d" % (qid, facet_id))
                    output_arr.append(";".join(["%s:%d" % (doc_no, label) for doc_no, label in cur_qfacet_posd_list]))
                    output_arr.append(";".join(cur_qfacet_candi_list))
                    line = "\t".join(output_arr)
                    if qid % k == test_i:
                        ftest.write("%s\n" % line)
                    elif qid % k == dev_i:
                        fdev.write("%s\n" % line)
                    else:
                        ftrain.write("%s\n" % line)

def output_bert_format(text_json_file, save_file):
    count = 0
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    with gzip.open(text_json_file, 'rt') as fin:
        content_dict = json.load(fin)
        for rid in content_dict:
            content_tokens = tokenizer.tokenize(content_dict[rid])
            content_tokens = content_tokens[:500] # cutoff documents longer than 500
            # also consider using concatenated top ranked passages instead
            content_dict[rid] = tokenizer.convert_tokens_to_ids(content_tokens)
            count += 1
            if count % 500 == 0:
                print("%d ids has been parsed!" % count)
    torch.save(content_dict, save_file)

def output_doc_psg_bert_format(text_json_file, doc_top_psg_file, save_file):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    qdoc_psg_dic = coll.defaultdict(dict)
    doc_id_set = set()
    count = 0
    with gzip.open(doc_top_psg_file, "rt") as fin:
        for line in fin:
            segs = line.strip("\r\n").split("\t")
            query, doc = segs[0], segs[1]
            psgs = segs[2].split(";")
            psg_text = " [SEP] ".join([x.split(":")[2] for x in psgs])
            content_tokens = tokenizer.tokenize(psg_text)
            content_tokens = content_tokens[:500] # cutoff documents longer than 500
            qdoc_psg_dic[query][doc] = tokenizer.convert_tokens_to_ids(content_tokens)
            doc_id_set.add(doc)
            count += 1
            if count % 500 == 0:
                print("%d ids has been parsed!" % count)
    count = 0
    with gzip.open(text_json_file, 'rt') as fin:
        content_dict = json.load(fin)
        for rid in content_dict:
            if rid in doc_id_set:
                continue
            content_tokens = tokenizer.tokenize(content_dict[rid])
            content_tokens = content_tokens[:300] # cutoff documents longer than 300
            # also consider using concatenated top ranked passages instead
            qdoc_psg_dic["None"][rid] = tokenizer.convert_tokens_to_ids(content_tokens)
            # relevant docs that are not in the top 50 ranklists. 
            count += 1
            if count % 500 == 0:
                print("%d ids has been parsed!" % count)
    torch.save(qdoc_psg_dic, save_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default="/net/home/kbi/ingham_disk/conv_search/qulac/data", type=str)
    args = parser.parse_args()
    data_path = os.path.join(args.root_dir, "galago_index")
    output_path = os.path.join(args.root_dir, "working")
    doc_dir = os.path.join(args.root_dir, "clean_topic_docs")
    # rank_file = os.path.join(data_path, "clarify_q_init_doc.mu1500.ranklist")
    rank_file = os.path.join(data_path, "cq_top_doc_rerank100.ranklist")
    qrel_file = os.path.join(args.root_dir, "qrels/multifacet.qrels.pos.txt")
    topic_ranklist_dic = coll.defaultdict(list)
    read_galago_ranklist(rank_file, topic_ranklist_dic)
    qrel_dict = read_qrels(qrel_file)
    doc_set = set()
    for qid in qrel_dict:
        for facet_id in qrel_dict[qid]:
            doc_set = doc_set.union(set([x for x,y in qrel_dict[qid][facet_id]]))
    for qid in topic_ranklist_dic:
        doc_set = doc_set.union(set(topic_ranklist_dic[qid]))

    # output_pos_and_candidate_doc(doc_dir, output_path, doc_set)
    partition_to_k_folds(qrel_dict, topic_ranklist_dic, output_path)

    rank_file = os.path.join(data_path, "clarify_q_init_q.ranklist")
    qrel_file = os.path.join(args.root_dir, "qrels/questions.labels.txt")
    topic_ranklist_dic = coll.defaultdict(list)
    read_galago_ranklist(rank_file, topic_ranklist_dic)
    qrel_dict = read_qrels(qrel_file)

    qulac_path = os.path.join(args.root_dir, "qulac/qulac.json")
    # output_question_text(qulac_path, output_path)
    partition_to_k_folds(qrel_dict, topic_ranklist_dic, output_path, is_q=True)
    sys.exit(0)
    doc_dict_file = "%s/candi_doc_dict.json.gz" % (output_path)
    cq_dict_file = "%s/questions.json.gz" % output_path
    bert_doc_file = os.path.join(output_path, "candi_doc_dict.pt")
    bert_cq_file = os.path.join(output_path, "questions_dict.pt")
    # output_bert_format(cq_dict_file, bert_cq_file)
    # output_bert_format(doc_dict_file, bert_doc_file)
    bert_doc_file = os.path.join(output_path, "candi_doc_psg_dict.pt")
    doc_top_psg_file = os.path.join(data_path, "topic.top_doc_psg.txt.gz")
    # output_doc_psg_bert_format(doc_dict_file, doc_top_psg_file, bert_doc_file)
