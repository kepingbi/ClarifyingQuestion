import os
import gzip
import argparse
import nltk


def read_content_from_trec_file(trect_file):
    content_dic = dict()
    with gzip.open(trect_file, "rt") as fin:
        for line in fin:
            if line.startswith("<DOCNO>"):
                doc_no = line.strip("\r\n")[len("<DOCNO>"):-len("</DOCNO>")]
            if line.startswith("<TEXT>"):
                text = line.strip("\r\n")[len("<TEXT>"):-len("</TEXT>")]
                if doc_no not in content_dic:
                    content_dic[doc_no] = text
    return content_dic

def output_doc_in_text(content_dic:dict(), output_dir):
    sent_tokenizer = nltk.tokenize.PunktSentenceTokenizer()
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "ql_top_trec_doc.txt")
    with open(output_file, "w") as fout:
        for doc_no in content_dic:
            text = content_dic[doc_no].strip("\r\n")
            sentences = sent_tokenizer.tokenize(text)
            for sent in sentences:
                fout.write("%s\n" % sent)
            fout.write("\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="/net/home/kbi/ingham_disk/conv_search/qulac/data/galago_index", type=str)
    parser.add_argument('--output_path', default="/net/home/kbi/ingham_disk/conv_search/qulac/data/bert_pretrain", type=str)
    
    args = parser.parse_args()
    trec_file = os.path.join(args.data_path, 'clueweb_doc.trectext.gz')
    content_dic = read_content_from_trec_file(trec_file)
    output_doc_in_text(content_dic, args.output_path)

