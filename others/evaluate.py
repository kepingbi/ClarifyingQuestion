import gzip
import os
import collections as coll
import math
import argparse
import sys
import numpy as np

def calc_idcg(iranklist):
    # iranklist = [1] * len(rel_set)
    #only label in the list
    rank = 1
    idcg_list = []
    for rank, label in enumerate(iranklist):
        rank += 1
        gain = (math.pow(2, label) - 1) / math.log((rank + 1), 2)
        idcg_list.append(gain)
    return idcg_list

def calc_dcg(ranklist):
    dcg_list = []
    for rank, label in enumerate(ranklist):
        rank += 1
        gain = 0.0
        if label > 0:
            gain = (math.pow(2, label) - 1) / math.log((rank + 1), 2)
        dcg_list.append(gain)
    return dcg_list

def calc_ndcg(ranklist, iranklist, pos=10):
    iranklist = iranklist[:pos]
    ranklist = ranklist[:pos]
    dcg = 0.0
    ndcg = 0.0
    gain = 0.0
    idcg = sum(calc_idcg(iranklist))
    if idcg == 0:
        return 0.
    for rank, label in enumerate(ranklist):
        rank += 1
        if label > 0:
            #print rank, label
            gain += (math.pow(2, label) - 1) / math.log((rank + 1), 2)
    dcg = gain
    ndcg = dcg / idcg
    #print dcg, idcg, ndcg
    return ndcg

def calc_map(ranklist, rel_set, pos=1000):
    ranklist = ranklist[:pos]
    val = 0.0
    rel_count = 0
    for i in range(len(ranklist)):
        rank = i + 1
        if ranklist[i] > 0:
            rel_count += 1
            val += rel_count / (rank + 0.0)
    val /= len(rel_set)
    return val

def calc_err(ranklist, mlabel=1):
    ''' maximum grade: 4 if there are 5 grades (0,1,2,3,4) in total
    '''
    err_list = []
    left_r = 1.0
    for rank, label in enumerate(ranklist):
        rank += 1
        gain = 0.0
        if label > 0:
            r = (math.pow(2, label) - 1) / math.pow(2, mlabel)
            #print rank, r, left_r, left_r*r/rank
            gain = left_r*r/rank
            left_r *= (1-r)
        err_list.append(gain)
    return err_list

def calc_mrr(ranklist):
    for i, label in enumerate(ranklist):
        # if label > 0: # 
        if label > 1: # The first true question regarding the aspect 
            return 1.0 / (i + 1)
    return 0.0

def parse_eval_result(eval_res_str):
    result_dic = dict()
    lines = eval_res_str.strip().split('\n')
    #line[0] is the file name
    session_q_count = int(lines[1].split()[1])
    scored_session_q_count = int(lines[2].split()[1])
    for line in lines[3:]:
        metric, value = line.split()
        result_dic[metric] = float(value)
    return session_q_count, scored_session_q_count, result_dic

#read from a detail evalution file of session, q
def read_detail_eval_result(eval_file, rank_cut_offs):
    session_q_ndcg = [coll.defaultdict(float) for _ in range(1 + len(rank_cut_offs))]
    session_q_precision = [coll.defaultdict(float) for _ in range(1 + len(rank_cut_offs))]
    session_q_err = [coll.defaultdict(float) for _ in range(1 + len(rank_cut_offs))]
    session_q_map = coll.defaultdict(float)
    session_q_mrr = coll.defaultdict(float)
    prev_qcount_dic = coll.defaultdict(int)

    scored_session_q_count = 0
    metric_col_id_dic = dict()
    with open(eval_file, 'r') as feval:
        segs = feval.readline().strip('\r\n').split('\t')[2].split()
        for i in range(len(segs)):
            metric_col_id_dic[segs[i]] = i
        for line in feval:
            line = line.strip('\r\n')
            segs = line.split(' ')
            if len(segs) < len(rank_cut_offs) * 3:  # ndcg, precision, err
                continue
            segs = line.split('\t')
            qid, prev_qcount = int(segs[0]), int(segs[1])
            prev_qcount_dic[qid] = prev_qcount
            perf_values = [float(x) for x in segs[2].split(' ')]
            session_q_map[qid] = perf_values[metric_col_id_dic['MAP@50']]
            session_q_mrr[qid] = perf_values[metric_col_id_dic['MRR']]
            cut_off_strs = ['@%d' % x for x in rank_cut_offs] + ['']
            for i, cutoff in enumerate(cut_off_strs):
                session_q_ndcg[i][qid] = perf_values[metric_col_id_dic['NDCG%s'%cutoff]]
                session_q_err[i][qid] = perf_values[metric_col_id_dic['ERR%s'%cutoff]]
                session_q_precision[i][qid] = perf_values[metric_col_id_dic['PRECISION%s'%cutoff]]
            scored_session_q_count += 1

    if not session_q_mrr:
        print("None valid session query pair!")
        sys.exit(-1)
    session_q_num = len(session_q_mrr)
    return session_q_num, scored_session_q_count, prev_qcount_dic, \
        session_q_mrr, session_q_map, session_q_ndcg, session_q_precision, session_q_err

# def read_qrels(qrels_file):
#     qrels_dic = dict()
#     with gzip.open(qrels_file, 'r') as fqrel:
#         for line in fqrel:
#             line = line.strip('\n')
#             segs = line.split('\t')
#             session = segs[0]
#             qid = int(segs[1])
#             purchases = segs[2].split(',')
#             qrels_dic[qid] = set(purchases)
#     return qrels_dic

def eval_rankfile_detail(rank_file, rank_cut_offs):
    session_q_ndcg = [coll.defaultdict(float) for _ in range(1 + len(rank_cut_offs))]
    session_q_precision = [coll.defaultdict(float) for _ in range(1 + len(rank_cut_offs))]
    session_q_err = [coll.defaultdict(float) for _ in range(1 + len(rank_cut_offs))]
    session_q_map = coll.defaultdict(float)
    session_q_mrr = coll.defaultdict(float)
    prev_qcount_dic = coll.defaultdict(int)

    scored_session_q_count = 0
    with gzip.open(rank_file, 'rt') as frank:
        for line in frank:
            line = line.strip('\n')
            segs = line.split('\t')
            query_infos = segs[0].split('@')
            qid = int(query_infos[0])
            uid = query_infos[1]
            prev_qcount = int(query_infos[2])
            # maybe len(query_infos) > 3
            prev_qcount_dic[qid] = prev_qcount
            ranklist = segs[1].split(';')
            doc_ratings = [int(x.split(':')[1]) for x in ranklist]
            doc_hit = [1 if x > 0 else 0 for x in doc_ratings]
            scored_session_q_count += 1
            dcg_list = calc_dcg(doc_ratings)
            sorted_pos_doc_ratings = [x for x in doc_ratings if x > 0]
            if len(sorted_pos_doc_ratings) == 0:
                continue
            sorted_pos_doc_ratings.sort(reverse=True)
            idcg_list = calc_idcg(sorted_pos_doc_ratings)
            err_list = calc_err(doc_ratings, mlabel=4)
            for i, cutoff in enumerate(rank_cut_offs):
                session_q_ndcg[i][qid] = sum(dcg_list[:cutoff]) / sum(idcg_list[:cutoff])
                session_q_err[i][qid] = sum(err_list[:cutoff])
                session_q_precision[i][qid] = \
                    (0.0 + sum(doc_hit[:cutoff])) / len(doc_ratings[:cutoff])

            session_q_ndcg[-1][qid] = sum(dcg_list) / sum(idcg_list)
            session_q_err[-1][qid] = sum(err_list)
            session_q_precision[-1][qid] = (0.0 + sum(doc_hit)) / len(doc_ratings)

            session_q_mrr[qid] = calc_mrr(doc_ratings)
            session_q_map[qid] = calc_map(doc_ratings, doc_ratings, 100)
            #if session == '1014564':
            # same session, query appear twice
            #    print doc_ratings
    if not session_q_mrr:
        print("None valid session query pair!")
        sys.exit(-1)
    session_q_num = len(session_q_mrr)

    #print '#session-q', session_q_num
    #print '#scored-session-q', scored_session_q_count
    return session_q_num, scored_session_q_count, prev_qcount_dic, \
        session_q_mrr, session_q_map, session_q_ndcg, session_q_precision, session_q_err

def accumulate_result_by_part(rank_cut_offs, session_q_ndcg, query_list):
    ndcg = [0.0] * (1+len(rank_cut_offs))
    #query_list = session_q_ndcg.keys() if not query_list else query_list
    session_q_num = len(query_list)
    for i in range(len(rank_cut_offs) + 1):
        ndcg[i] = sum([session_q_ndcg[i][qid] for qid in query_list]) / session_q_num
    metrics = ['NDCG@%d' % cutoff for cutoff in rank_cut_offs]
    metrics += ['NDCG']
    numbers = ['%f' % x for x in ndcg]
    return metrics, numbers


def accumulate_result(rank_cut_offs, session_q_mrr, session_q_map, \
        session_q_ndcg, session_q_precision, session_q_err):
    session_q_num = len(session_q_mrr)
    rank_map = 0.0
    ndcg = [0.0] * (1+len(rank_cut_offs))
    precision = [0.0] * (1+len(rank_cut_offs))
    err = [0.0] * (1+len(rank_cut_offs))
    mrr = 0.0
    for i in range(len(rank_cut_offs) + 1):
        ndcg[i] = sum(session_q_ndcg[i].values()) / session_q_num
        precision[i] = sum(session_q_precision[i].values()) / session_q_num
        err[i] = sum(session_q_err[i].values()) / session_q_num
    mrr = sum(session_q_mrr.values()) / session_q_num
    rank_map = sum(session_q_map.values()) / session_q_num
    metrics = ['MAP@50', 'MRR']
    metrics += ['NDCG@%d' % cutoff for cutoff in rank_cut_offs]
    metrics += ['NDCG']
    metrics += ['PRECISION@%d' % cutoff for cutoff in rank_cut_offs]
    metrics += ['PRECISION']
    metrics += ['ERR@%d' % cutoff for cutoff in rank_cut_offs]
    metrics += ['ERR']
    numbers = [str(rank_map), str(mrr)]
    numbers += ['%f' % x for x in ndcg]
    numbers += ['%f' % x for x in precision]
    numbers += ['%f' % x for x in err]
    return metrics, numbers

def eval_rankfile(para, rank_cut_offs, detail=True):
    if os.path.exists(para.from_file):
        session_q_count, scored_session_q_count, prev_qcount_dic, session_q_mrr, \
                session_q_map, session_q_ndcg, session_q_precision, session_q_err \
                    = read_detail_eval_result(para.from_file, rank_cut_offs)
    else:
        session_q_count, scored_session_q_count, prev_qcount_dic, session_q_mrr, \
                session_q_map, session_q_ndcg, session_q_precision, session_q_err \
                    = eval_rankfile_detail(para.rank_file, rank_cut_offs)
    #output detailed statistics

    if detail:
        with open(para.eval_file, 'w') as fout:
            session_q_list = sorted(session_q_mrr.keys()) # qids
            metrics = ['MAP@50', 'MRR']
            metrics += ['NDCG@%d' % cutoff for cutoff in rank_cut_offs]
            metrics += ['NDCG']
            metrics += ['PRECISION@%d' % cutoff for cutoff in rank_cut_offs]
            metrics += ['PRECISION']
            metrics += ['ERR@%d' % cutoff for cutoff in rank_cut_offs]
            metrics += ['ERR']
            fout.write("QID\tPREVQCOUNT\t%s\n" % (' '.join(metrics)))
            for qid in session_q_list:
                number_str = [str(session_q_map[qid]), str(session_q_mrr[qid])]
                number_str += ['%f' % x[qid] for x in session_q_ndcg]
                number_str += ['%f' % x[qid] for x in session_q_precision]
                number_str += ['%f' % x[qid] for x in session_q_err]
                fout.write("%s\t%s\t%s\n" % (qid, prev_qcount_dic[qid], ' '.join(number_str)))

    metrics, numbers = accumulate_result(rank_cut_offs, session_q_mrr, session_q_map, \
        session_q_ndcg, session_q_precision, session_q_err)
    print('#session-q\t%d' % session_q_count)
    print('#scored-session-q\t%d' % scored_session_q_count)
    result_dic = dict()
    for x, y in zip(metrics, numbers):
        print("%s\t%s" % (x, y))
        result_dic[x] = y
    return result_dic

RANK_CUT_OFFS = [1, 5, 10] #, 20, 50]

def str2bool(val):
    ''' parse bool type input parameters
    '''
    if val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def compare(para, rank_cut_offs, detail=True):
    context_q_count, scored_context_q_count, prev_qcount_dic, context_q_mrr, \
            context_q_map, context_q_ndcg, context_q_precision, context_q_err \
                = read_detail_eval_result(para.qcount_file, rank_cut_offs)
    base_q_count, scored_base_q_count, _, base_q_mrr, \
                base_q_map, base_q_ndcg, base_q_precision, base_q_err \
                    = read_detail_eval_result(para.from_file, rank_cut_offs)
    prev_qcount_qid_dic = coll.defaultdict(list)
    max_qcount = 0
    #print(len(prev_qcount_dic))
    for qid in prev_qcount_dic:
        max_qcount = max(max_qcount, prev_qcount_dic[qid])
        prev_qcount_qid_dic[prev_qcount_dic[qid]].append(qid)
    print([(x, len(prev_qcount_qid_dic[x])) for x in prev_qcount_qid_dic])
    result_dic = coll.defaultdict(dict)
    print("#PrevQcount\t#Queries\tMetrics\tBase\tContextModel")
    for count in range(max_qcount + 1):
        metrics, base_numbers = accumulate_result_by_part(
            rank_cut_offs, base_q_ndcg, prev_qcount_qid_dic[count])
        _, context_numbers = accumulate_result_by_part(
            rank_cut_offs, context_q_ndcg, prev_qcount_qid_dic[count])
        for x, y, z in zip(metrics, base_numbers, context_numbers):
            if not x == "NDCG@5":
                continue
            print("%d\t%d\t%s\t%s\t%s" % (
                count, len(prev_qcount_qid_dic[count]), x, y, z))
            result_dic[count][x] = (y, z)
    return result_dic

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--result_dir', \
        default="/home/keping2/EmailSearch/model/", help='ranklist')
    # parser.add_argument('-o', '--eval_file', \
    #     default="/home/keping2/EmailSearch/model/eval.txt", help='output')
    parser.add_argument('-i', '--from_file', \
        default="", help='output') # /home/keping2/EmailSearch/model/eval.txt
    parser.add_argument('--option', \
        default="eval", choices=["eval", "compare"], help='output')
    parser.add_argument('-q', '--qcount_file', \
        default="", help='output')
    parser.add_argument("--detail", type=str2bool, nargs='?', const=True, default=True, \
        help='show detail evaluation result for each record')
    para = parser.parse_args(sys.argv[1:])
    para.rank_file = os.path.join(para.result_dir, "test.best_model.ranklist.gz")
    print(para.rank_file)
    para.eval_file = os.path.join(para.result_dir, "eval.txt")
    if para.option == "eval":
        result_dic = eval_rankfile(para, RANK_CUT_OFFS, para.detail)
    elif para.option == "compare":
        compare(para, RANK_CUT_OFFS, para.detail)

    #for metric in sorted(result_dic.keys()):
    #    print metric, result_dic[metric]

if __name__ == '__main__':
    main()
