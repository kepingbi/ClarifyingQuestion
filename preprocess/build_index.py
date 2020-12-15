import unicodedata
import tarfile, gzip
import os
import sys
import argparse
import re
import json

def text_preprocess(text):
    new_text = re.sub(r'[^\x00-\x7f]',r' ', text)
    new_text = re.sub(r"\s+", " ", new_text)
    new_text = new_text.replace('\t',' ')
    new_text = new_text.replace('\n',' ')
    return(new_text)

def clean_files(dir_path, output_dir):
    # clean files and output to trec format for index building, merge duplicates. 
    # store the clean files to new files, and keep the mapping between doc_no, file_name
    os.makedirs(output_dir, exist_ok=True)
    for x in os.listdir(dir_path)[:1]:
        if not x.endswith('gz'):
            continue
        print(x)
        outname = os.path.join(output_dir, x.rsplit('.', maxsplit=2)[0]+'.gz')
        fname = os.path.join(dir_path, x)
        with tarfile.open(fname, "r:gz") as tar, gzip.open(outname, 'wt') as fout:
            for member in tar.getmembers():
                f = tar.extractfile(member)
                if f is None:
                    continue
                topic_dict = json.loads(f.read())
                for doc_id in topic_dict['id']:
                    content = topic_dict['text'][doc_id]
                    clean_content = text_preprocess(content)
                    topic_dict['text'][doc_id] = clean_content
                json.dump(topic_dict, fout)

# def output_doc_for_separate_topic(dir_path, output_dir):
#     new_output_dir = os.path.join(output_dir, 'clueweb_doc_trectext')
#     os.makedirs(new_output_dir, exist_ok=True)
#     doc_id_set = set()
#     for x in os.listdir(dir_path):
#         if not x.endswith('gz'):
#             continue
#         print(x)
#         outname = "%s/%s.trectext.gz" % (new_output_dir, x.split('.')[0])
#         fname = os.path.join(dir_path, x)
#         with gzip.open(fname, 'rt') as fin, gzip.open(outname, 'wt') as fout:
#             topic_dict = json.load(fin)
#             for doc_no in topic_dict['id']:
#                 cluebweb_id = topic_dict['id'][doc_no]
#                 if cluebweb_id in doc_id_set:
#                     continue
#                 doc_id_set.add(cluebweb_id)
#                 content = topic_dict['text'][doc_no]
#                 line = "<DOC>\n<DOCNO>%s</DOCNO>\n<TEXT>%s</TEXT>\n</DOC>\n" % (cluebweb_id, content)
#                 fout.write(line)

# def build_separate_index(output_dir):
#     data_path = os.path.join(output_dir, 'clueweb_doc_trectext')
#     new_output_dir = os.path.join(output_dir, 'clueweb_doc_index')
#     new_output_dir = "/home/kbi/work1kbi/clarify_question/working/galago_index"
#     new_data_path = os.path.join(new_output_dir, 'clueweb_doc_trectext')
#     new_output_dir = os.path.join(new_output_dir, 'clueweb_doc_index')
#     fscript = open("build_index.sh", 'w') # created for running on sydney
#     for x in os.listdir(data_path):
#         if not x.endswith('gz'):
#             continue
#         print(x)
#         fname = os.path.join(new_data_path, x)
#         index_dir = "%s/%s_index" % (new_output_dir, x.split('.')[0])
#         cmd = "srun galago build --indexPath=%s --inputPath+%s &\n" % (index_dir, fname)
#         fscript.write(cmd)
#     fscript.close()

def output_questions_as_trec(data_path, output_path):
    output_file = os.path.join(output_path, "clarify_questions.trectext.gz")
    with open(data_path) as fin, gzip.open(output_file, "wt") as fout:
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
            if global_qid in qid_set:
                continue
            qid_set.add(global_qid)
            qtext = text_preprocess(questions[e_id])
            line = "<DOC>\n<DOCNO>%s</DOCNO>\n<TEXT>%s</TEXT>\n</DOC>\n" % (global_qid, qtext)
            fout.write(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="/net/home/kbi/ingham_disk/conv_search/qulac/data/qulac_topic_docs", type=str)
    parser.add_argument('--output_path', default="/net/home/kbi/ingham_disk/conv_search/qulac/data/clean_topic_docs", type=str)
    parser.add_argument('--cq_path', default="/net/home/kbi/projects/conv_search/qulac/data/qulac/new_qulac.json", type=str)
    args = parser.parse_args()
    trec_path = os.path.join(os.path.dirname(args.data_path), "galago_index")
    # clean_files(args.data_path, args.output_path)
    # output_questions_as_trec(args.cq_path, trec_path)
    # output_doc_for_separate_topic(args.output_path, trec_path)
    ## we cannot build separate index for each topic, query words will have low idf.
    # build_separate_index(trec_path)
