import os
import sys
import json
import argparse
import collections as coll
import gzip
import re
import nltk

def prepare_galago_json_for_query_faceted(data_path, output_path, index_dir, alpha=0.5):
    os.makedirs(output_path, exist_ok=True)
    request_json_dict = {"index": index_dir}
    request_json_dict["requested"] = 50
    request_json_dict["queries"] = []
    with open(data_path) as fin:
        fjson = json.load(fin)
        entry_ids = fjson["topic_id"].keys()
        question_ids = fjson["topic_facet_question_id"]
        qid_set = set()
        for e_id in entry_ids:
            topic_id, facet_id, _ = question_ids[e_id].split("-")
            #qid = '0' if qid == "X" else qid # do original query
            global_facet_id = "%s-%s" % (topic_id, facet_id)
            if global_facet_id in qid_set:
                continue
            qid_set.add(global_facet_id)
            qstr = fjson["topic"][e_id]
            curq_dic = {"number": global_facet_id, "text": qstr}
            request_json_dict["queries"].append(curq_dic)
    output_file = os.path.join(output_path, "request_clarify_q_facet.json")
    with open(output_file, "wt") as fout:
        json.dump(request_json_dict, fout, indent=4)

# get json file of queries, queries+questions
# retrieve documents / clarifying questions by running the json file on the indices.  
def prepare_galago_json_for_query(data_path, output_path, index_dir, alpha=0.5):
    os.makedirs(output_path, exist_ok=True)
    request_json_dict = {"index": index_dir}
    request_json_dict["requested"] = 50
    request_json_dict["queries"] = []
    with open(data_path) as fin:
        fjson = json.load(fin)
        entry_ids = fjson["topic_id"].keys()
        question_ids = fjson["topic_facet_question_id"]
        questions = fjson["question"]
        qid_set = set()
        for e_id in entry_ids:
            topic_id, _, qid = question_ids[e_id].split("-")
            #qid = '0' if qid == "X" else qid # do original query
            global_qid = "%s-%s" % (topic_id, qid)
            if global_qid in qid_set:
                continue
            qid_set.add(global_qid)
            cqtext = questions[e_id]
            query = fjson["topic"][e_id]
            if cqtext == "":
                qstr = "#combine(%s)" % (query)
            else:
                qstr = "#combine:0=%.1f:1=%.1f(#combine(%s) #combine(%s))" % (alpha, (1-alpha), query, cqtext)
            curq_dic = {"number": global_qid, "text": qstr}
            #combine:0=0.4:1=0.6(#combine(soviet withdrawal) #combine:0=0.4:1=0.6(soviets troops))
            request_json_dict["queries"].append(curq_dic)
    output_file = os.path.join(output_path, "request_clarify_q.json")
    with open(output_file, "wt") as fout:
        json.dump(request_json_dict, fout, indent=4)

# def prepare_galago_json(data_path, output_path, index_dir, alpha=0.5):
#     os.makedirs(output_path, exist_ok=True)
#     request_json_dict = dict()
#     with open(data_path) as fin:
#         fjson = json.load(fin)
#         entry_ids = fjson["topic_id"].keys()
#         question_ids = fjson["topic_facet_question_id"]
#         questions = fjson["question"]
#         qid_set = set()
#         for e_id in entry_ids:
#             topic_id, _, qid = question_ids[e_id].split("-")
#             #qid = '0' if qid == "X" else qid # do original query
#             global_qid = "%s-%s" % (topic_id, qid)
#             if global_qid in qid_set:
#                 continue
#             qid_set.add(global_qid)
#             cqtext = questions[e_id]
#             query = fjson["topic"][e_id]
#             if topic_id not in request_json_dict:
#                 request_json_dict[topic_id] = {"index": "%s/%s_index" % (index_dir, topic_id)}
#                 request_json_dict[topic_id]["requested"] = 50
#                 request_json_dict[topic_id]["queries"] = []
#             if cqtext == "":
#                 qstr = "#combine(%s)" % (query)
#             else:
#                 qstr = "#combine:0=%.1f:1=%.1f(#combine(%s) #combine(%s))" % (alpha, (1-alpha), query, cqtext)
#             curq_dic = {"number": global_qid, "text": qstr}
#             #combine:0=0.4:1=0.6(#combine(soviet withdrawal) #combine:0=0.4:1=0.6(soviets troops))
#             request_json_dict[topic_id]["queries"].append(curq_dic)
#     with open("search.sh", "w") as fscript:
#         for topic_id in request_json_dict:
#             output_file = os.path.join(output_path, "%s.json" % topic_id)
#             with open(output_file, "wt") as fout:
#                 json.dump(request_json_dict[topic_id], fout, indent=4)
#             output_file = os.path.join(
#                 os.path.dirname(index_dir), "request_json", "%s.json" % topic_id)
#             # for syndey
#             rank_file = os.path.join(
#                 os.path.dirname(index_dir), "doc_ranklists", "%s.ranklist" % topic_id)
#             line = "srun galago batch-search %s &> %s &\n" % (output_file, rank_file)
#             fscript.write(line)

def read_galago_ranklist(rank_file, topic_ranklist_dic):
    with open(rank_file, 'r') as frank:
        for line in frank:
            segs = line.strip('\r\n').split(' ')
            if not segs[0].endswith("X"):
                continue
            doc_id = segs[2]
            topic_ranklist_dic[segs[0]].append(doc_id)

# def read_galago_ranklist_dir(rank_dir):
#     # read doc id ranklist for the original query. 
#     # 101-X Q0 clueweb09-en0008-78-08029 17 -4.75516188 galago
#     topic_ranklist_dic = coll.defaultdict(list)
#     for x in os.listdir(rank_dir):
#         rank_file = os.path.join(rank_dir, x)
#         read_galago_ranklist(rank_file, topic_ranklist_dic)
#     return topic_ranklist_dic

def output_doc_as_trec(rank_file, doc_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    outname = os.path.join(output_dir, 'clueweb_doc.trectext.gz')
    topic_ranklist_dic = coll.defaultdict(list)
    read_galago_ranklist(rank_file, topic_ranklist_dic)
    doc_id_set = set()
    with gzip.open(outname, 'wt') as fout:
        for x in os.listdir(doc_path):
            if not x.endswith('gz'):
                continue
            print(x)
            fname = os.path.join(doc_path, x)
            topic = x.split('.')[0] # 1,2,...200 except 95 and 100
            cur_doc_set = set(topic_ranklist_dic["%s-X" % topic])
            with gzip.open(fname, 'rt') as fin:
                topic_dict = json.load(fin)
                for doc_no in topic_dict['id']:
                    cluebweb_id = topic_dict['id'][doc_no]
                    if cluebweb_id not in cur_doc_set:
                        continue
                    if cluebweb_id in doc_id_set:
                        continue
                    doc_id_set.add(cluebweb_id)
                    content = topic_dict['text'][doc_no]
                    line = "<DOC>\n<DOCNO>%s</DOCNO>\n<TEXT>%s</TEXT>\n</DOC>\n" % (cluebweb_id, content)
                    fout.write(line)

def output_top_doc_psg_as_trec(rank_file, doc_dir, output_dir, psg_length=100, shift_size=75):
    ''' rank_dir: top 50 documents for each topic; 
        doc_dir: candidate documen text for each topic
        output_dir: output the trectext path
    '''
    os.makedirs(output_dir, exist_ok=True)
    outname = os.path.join(output_dir, 'clueweb_top_doc_psg.trectext.gz')
    topic_ranklist_dic = coll.defaultdict(list)
    read_galago_ranklist(rank_file, topic_ranklist_dic)
    doc_id_set = set()
    sent_tokenizer = nltk.tokenize.PunktSentenceTokenizer()
    with gzip.open(outname, 'wt') as fout:
        for x in os.listdir(doc_dir):
            if not x.endswith('gz'):
                continue
            print(x)
            fname = os.path.join(doc_dir, x)
            topic = x.split('.')[0] # 1,2,...200 except 95 and 100
            cur_doc_set = set(topic_ranklist_dic["%s-X" % topic])
            with gzip.open(fname, 'rt') as fin:
                topic_dict = json.load(fin)
                for doc_no in topic_dict['id']:
                    cluebweb_id = topic_dict['id'][doc_no]
                    if cluebweb_id not in cur_doc_set:
                        continue
                    if cluebweb_id in doc_id_set:
                        continue
                    doc_id_set.add(cluebweb_id)
                    psg_no = 0
                    text = topic_dict['text'][doc_no]
                    sentences = sent_tokenizer.tokenize(text)
                    token_sents = []
                    for sent in sentences:
                        words = sent.split()
                        if len(words) > psg_length:
                            for i in range(0, len(words), psg_length):
                                token_sents.append(words[i:i+psg_length])
                        else:
                            token_sents.append(words)
                    i = 0
                    psg_text = []
                    while i <= len(token_sents):
                        if i == len(token_sents) or len(psg_text) + len(token_sents[i]) > psg_length:
                            psg_content = " ".join(psg_text)
                            psg_id = "%s-psg%d" % (cluebweb_id, psg_no)
                            psg_no += 1
                            line = "<DOC>\n<DOCNO>%s</DOCNO>\n<TEXT>%s</TEXT>\n</DOC>\n" \
                                % (psg_id, psg_content)
                            fout.write(line)
                            psg_text = []
                            if i == len(token_sents):
                                break
                            else:
                                continue
                        psg_text.extend(token_sents[i])
                        i += 1
                    # words = topic_dict['text'][doc_no].split()
                    # for start_pos in range(0, len(words), shift_size):
                    #     psg_content = " ".join(words[start_pos:start_pos+psg_length])
                    #     psg_id = "%s-psg%d" % (cluebweb_id, psg_no)
                    #     psg_no += 1
                    #     line = "<DOC>\n<DOCNO>%s</DOCNO>\n<TEXT>%s</TEXT>\n</DOC>\n" \
                    #         % (psg_id, psg_content)
                    #     fout.write(line)

def prepare_galago_psg_json(data_path, output_dir, index_dir, mu=100):
    os.makedirs(output_dir, exist_ok=True)
    request_json_dict = {"index": index_dir}
    request_json_dict["requested"] = 1000
    request_json_dict["mu"] = 100
    request_json_dict["queries"] = []
    with open(data_path) as fin:
        fjson = json.load(fin)
        entry_ids = fjson["topic_id"].keys()
        question_ids = fjson["topic_facet_question_id"]
        qid_set = set()
        for e_id in entry_ids:
            topic_id, _, qid = question_ids[e_id].split("-")
            if not qid == "X":
                continue
            query = fjson["topic"][e_id]
            global_qid = "%s-%s" % (topic_id, qid)
            if global_qid in qid_set:
                continue
            qid_set.add(global_qid)
            qstr = "#combine(%s)" % (query)
            curq_dic = {"number": global_qid, "text": qstr}
            request_json_dict["queries"].append(curq_dic)
    with open(os.path.join(output_dir, "passage_query.json"), 'w') as fout:
        json.dump(request_json_dict, fout, indent=4)

def clean(text):
    return re.sub(r'[^a-zA-Z0-9 ]', '', text)

def read_content_from_trec_file(trect_file, lowercase=True):
    content_dic = dict()
    with gzip.open(trect_file, "rt") as fin:
        for line in fin:
            if line.startswith("<DOCNO>"):
                doc_no = line.strip("\r\n")[len("<DOCNO>"):-len("</DOCNO>")]
            if line.startswith("<TEXT>"):
                text = line.strip("\r\n")[len("<TEXT>"):-len("</TEXT>")]
                if lowercase:
                    text = text.lower()
                if doc_no not in content_dic:
                    content_dic[doc_no] = clean(text)
    return content_dic

def get_top_psg_for_top_doc(psg_ranklist, psg_trec_file, doc_ranklist_file, topk=3):
    '''psg_ranklist: passage ranklists which is a single file
       psg_trec_file: the content of each psg
       doc_rank_dir: the directory that contains document ranklists of each topic 
       # query_id\ttop_doc_id1:passages_id:ranking:content passages_id:ranking:content
       # query_id\ttop_doc_id2:passages_id:ranking:content passages_id:ranking:content
    '''
    output_file = os.path.join(os.path.dirname(psg_ranklist), "topic.top_doc_psg.txt.gz")
    psg_content_dic = read_content_from_trec_file(psg_trec_file)
    psg_ranklist_dic = coll.defaultdict(list)
    read_galago_ranklist(psg_ranklist, psg_ranklist_dic)
    doc_ranklist_dic = coll.defaultdict(list)
    read_galago_ranklist(doc_ranklist_file, doc_ranklist_dic)
    # topic psg-id
    with gzip.open(output_file, 'wt') as fout:
        for i in range(1, 201):
            if i == 95 or i == 100:
                continue
            query_id = "%d-X" % i
            cur_doc_ranklist = doc_ranklist_dic[query_id]
            cur_psg_ranklist = psg_ranklist_dic[query_id]
            doc_psg_rank_dic = coll.defaultdict(list)
            for rank, psg_id in enumerate(cur_psg_ranklist):
                doc_id = psg_id.rsplit('-', maxsplit=1)[0]
                doc_psg_rank_dic[doc_id].append((psg_id, rank+1))

            for doc_id in cur_doc_ranklist: # from rank 1 to 50
                first_psg_id = "%s-psg0" % (doc_id) # the first passage
                kept_psg_list = [(first_psg_id, -1)]
                topk_psg_set = set([psg_id for psg_id, _ in doc_psg_rank_dic[doc_id][:topk]])
                if first_psg_id in topk_psg_set:
                    kept_psg_list = doc_psg_rank_dic[doc_id][:topk]
                else:
                    kept_psg_list.extend(doc_psg_rank_dic[doc_id][:topk-1])
                kept_psg_list.sort(key=lambda x:x[0])
                arr = ["%s:%d:%s" % (psg_id, rank, psg_content_dic[psg_id]) for psg_id, rank in kept_psg_list]
                line = "%s\t%s\t%s\n" % (query_id, doc_id, ";".join(arr))
                fout.write(line)

def read_ranklist_info(rank_file):
    topic_ranklist_dic = coll.defaultdict(list)
    with open(rank_file, 'r') as frank:
        for line in frank:
            segs = line.strip('\r\n').split(' ')
            doc_id = segs[2]
            rank = int(segs[3])
            score = float(segs[4])
            topic_ranklist_dic[segs[0]].append((doc_id, rank, score))
    return topic_ranklist_dic

def rerank_init_top(rank_file, outfile, topk=50):
    topic_ranklist_dic = read_ranklist_info(rank_file)
    filtered_ranklist_dic = dict()
    init_rankset_dic = dict()
    for qid in topic_ranklist_dic:
        # print(len(topic_ranklist_dic[qid]))
        if qid.endswith("X"):
            filtered_ranklist_dic[qid] = topic_ranklist_dic[qid][:topk]
            init_rankset_dic[qid] = set([x for x,y,z in topic_ranklist_dic[qid][:topk]])
    for qid in topic_ranklist_dic:
        topic_id, _ = qid.split('-')
        ori_qid = "%s-X" % topic_id
        filtered_list = [x for x in topic_ranklist_dic[qid] if x[0] in init_rankset_dic[ori_qid]]
        filtered_ranklist_dic[qid] = filtered_list
    
    with open(outfile, 'w') as fout:
        ordered_keys = sorted(filtered_ranklist_dic.keys())
        for qid in ordered_keys:
            for doc_id, rank, score in filtered_ranklist_dic[qid]:
                line = "%s Q0 %s %d %f\n" % (qid, doc_id, rank, score)
                fout.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="/net/home/kbi/ingham_disk/conv_search/qulac/data/galago_index", type=str)
    parser.add_argument('--index_path', default="/net/home/kbi/ingham_disk/conv_search/qulac/data/galago_index/clueweb_doc_index", type=str)
    parser.add_argument('--cq_path', default="/net/home/kbi/projects/conv_search/qulac/data/qulac/new_qulac.json", type=str)
    parser.add_argument('--doc_dir', default="/net/home/kbi/ingham_disk/conv_search/qulac/data/clean_topic_docs", type=str)
    
    args = parser.parse_args()
    request_json_path = os.path.join(args.data_path, "request_json")
    index_path = os.path.join(args.data_path, "clueweb_doc_index")
    index_path = "/home/kbi/work1kbi/clarify_question/working/galago_index/clueweb_doc_index" # for sydney
    # 1. issue queries to the separate index and collect top ranked clarifying questions
    index_path = os.path.join(args.data_path, "clarify_question_index")
    # prepare_galago_json_for_query(args.cq_path, args.data_path, index_path)
    # prepare_galago_json_for_query_faceted(args.cq_path, args.data_path, index_path)
    # sys.exit(0)
    # For queries on document index, use Mohammad's code to retrieve QL results. 
    # Call the QL function with /net/home/kbi/projects/conv_search/qulac/src/doc_initial_retrieval.py

    # 2. build index for passages in top ranked documents
    rank_file = os.path.join(args.data_path, "clarify_q_init_doc.mu1500.ranklist")
    # output_top_doc_psg_as_trec(rank_file, args.doc_dir, args.data_path)
    # output_doc_as_trec(rank_file, args.doc_dir, args.data_path)
    ## galago build --indexPath=clueweb_top_doc_index --inputPath+clueweb_doc.trectext.gz
    ## galago build --indexPath=clueweb_top_doc_psg_index --inputPath+clueweb_top_doc_psg.trectext.gz
    # 3. retrieve passages from the index
    index_path = os.path.join(args.data_path, "clueweb_top_doc_psg_index")
    # prepare_galago_psg_json(args.cq_path, args.data_path, index_path)
    # galago batch-search passage_query.json &> top_doc_passage.ranklist
    # 4. get the top 3/4 passages corresponding to each top retrieved document. 
    psg_ranklist = os.path.join(args.data_path, "top_doc_passage.ranklist")
    psg_trec_file = os.path.join(args.data_path, "clueweb_top_doc_psg.trectext.gz")
    get_top_psg_for_top_doc(psg_ranklist, psg_trec_file, rank_file)
    sys.exit(0)

    whole_ranklist = os.path.join(args.data_path, "cq_top_doc.ranklist")
    out_filtered_ranklist = os.path.join(args.data_path, "cq_top_doc_rerank50.ranklist")
    rerank_init_top(whole_ranklist, out_filtered_ranklist)


