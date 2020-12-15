import os
import sys
import json
import argparse
import collections as coll

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="/net/home/kbi/projects/conv_search/qulac/data/qulac/qulac.json", type=str)
    parser.add_argument('--output_path', default="/net/home/kbi/projects/conv_search/qulac/data/qrels", type=str)
    return parser.parse_args()

def cut_neg_answers(data_path, output_path):
    output_file = "%s/questions.labels.txt" % output_path
    pos_q_dic = coll.defaultdict(int)
    with open(data_path) as fin, open(output_file, "w") as fout:
        fjson = json.load(fin)
        entry_ids = fjson["topic_id"].keys()
        question_ids = fjson["topic_facet_question_id"]
        answers = fjson["answer"]
        for e_id in entry_ids:
            topic_id, facet_id, qid = question_ids[e_id].split("-")
            if qid == "X":
                continue
            global_qid = "%s-%s" % (topic_id, qid)
            ans = answers[e_id]
            label = 1
            topic_facet_id = "%s-%s" % (topic_id, facet_id)
            if ans.startswith("yes") or ans.startswith("sure"):
                label = 2
                pos_q_dic[topic_facet_id] += 1
            else:
                answers[e_id] = "no"
            line = "%s-%s Q0 %s %d\n" % (topic_id, facet_id, global_qid, label)
            fout.write(line)
        new_data = os.path.dirname(data_path) + "/new_qulac.json"
        json.dump(fjson, open(new_data, "w"))
        topic_facet_id_set = dict([(fjson["topic_facet_id"][e_id], e_id) for e_id in entry_ids])
        for tf_id in topic_facet_id_set:
            print("%s\t%s\t%d" % (topic_facet_id_set[tf_id], tf_id, pos_q_dic[tf_id]))

def main(args):
    cut_neg_answers(args.data_path, args.output_path)

if __name__ == '__main__':
    main(parse_args())

