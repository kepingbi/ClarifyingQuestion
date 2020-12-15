import os
from initial_retrieval import read_galago_ranklist

def output_ql_for_facet_evaluation(data_path, output_path):
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

