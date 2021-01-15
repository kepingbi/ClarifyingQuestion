import os
import sys
import json
import math
import argparse
from collections import defaultdict
import gzip
import nltk
import krovetz
ks = krovetz.PyKrovetzStemmer()

def collect_cq_term_idf(text_json_file, output_path):
    term_idf = defaultdict(int)
    with gzip.open(text_json_file, 'rt') as fin:
        content_dict = json.load(fin)
        for rid in content_dict:
            words = content_dict[rid].split()
            for w in words:
                term_idf[w] += 1
    print(len(content_dict))
    # print([(x, term_idf[x]) for x in sorted(term_idf, key=term_idf.get, reverse=True)])
    for w in term_idf:
        term_idf[w] = math.log(len(content_dict) / term_idf[w])
    with open(output_path, "w") as fout:
        for w in sorted(term_idf, key=term_idf.get, reverse=True):
            fout.write("%s %f\n" % (w, term_idf[w]))
    return content_dict, term_idf

def output_question_words(data_path, cq_text_dic, term_idf, output_path):
    high_idf_terms = set([x for x in term_idf if term_idf[x] > 3.784947]) # history(3.818848, 58), there(3.784947, 60), find(3.752157, 62)
    stopwords = nltk.corpus.stopwords.words('english')
    topic_dic = read_queries(data_path)
    cq_remain_dic = dict()
    for cq in cq_text_dic:
        topic_id = cq.split("-")[0]
        query_words = topic_dic[topic_id]
        cq_words = cq_text_dic[cq].split()
        words_remain = []
        for w in cq_words:
            if w in stopwords:
                continue
            if w not in high_idf_terms:
                continue
            if ks.stem(w) in query_words:
                continue
            words_remain.append((w, term_idf[w]))
        cq_remain_dic[cq] = words_remain

    with open(output_path, "wt") as fout:
        json.dump(cq_remain_dic, fout, indent=4)

def read_queries(qulac_file):
    topic_dic = dict() # query
    with open(qulac_file) as fin:
        fjson = json.load(fin)
        entry_ids = fjson["topic_id"].keys()
        topics = fjson["topic"]
        question_ids = fjson["topic_facet_question_id"]
        for e_id in entry_ids:
            topic_id, facet_id, qid = question_ids[e_id].split("-")
            if qid == "X":
                continue
            if topic_id not in topic_dic:
                tokens = topics[e_id].split()
                topic_dic[topic_id] = set([ks.stem(x) for x in tokens])
                
    return topic_dic


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cq_path', default="/mnt/scratch/kbi/conv_search/qulac/data/working/questions.json.gz", type=str)
    parser.add_argument('--idf_path', default="/net/home/kbi/projects/conv_search/qulac/data/working/cq_word_idf.txt", type=str)
    parser.add_argument('--qulac_path', default="/net/home/kbi/projects/conv_search/qulac/data/qulac/new_qulac.json", type=str)
    parser.add_argument('--out_cq_path', default="/net/home/kbi/projects/conv_search/qulac/data/working/imp_cq_words.json", type=str)
    
    args = parser.parse_args()
    cq_text_dic, term_idf = collect_cq_term_idf(args.cq_path, args.idf_path)
    output_question_words(args.qulac_path, cq_text_dic, term_idf, args.out_cq_path)
