import sys
import os
import argparse

#titanx-long #1080ti-long #2080ti-long   # Partition to submit to
#SBATCH --mem=96000    # Memory in MB per node allocated
config_str = """#!/bin/bash

#SBATCH --partition=titanx-short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=64000    # Memory in MB per node allocated
#SBATCH --ntasks-per-node=4
"""

WORKING_DIR="/mnt/nfs/work1/croft/kbi/conv_search/working/cq_ranker"
OUTPUT_DIR="/mnt/nfs/work1/croft/kbi/conv_search/output/cq_ranker"
BERT_PRETRAIN="/mnt/nfs/work1/croft/kbi/conv_search/working/bert_pretrain/finetuned_lm"

script_path = "python main.py"

para_names = ['fold', 'init_model', 'model_name', \
    'selector', 'rl', 'init', 'rerank', 'fix_scorer', 'min_hist_turn', 'max_hist_turn', \
        'decay_method', 'lr', 'warmup_steps', 'max_train_epoch', 'batch_size', 'candi_batch_size', \
        'inter_embed_size', 'rerank_topk', 'mode', 'init_cq'] #, 'eval_k', 'rank_cutoff', 'eval_pos']
short_names = ['fold', 'im', 'mn', 'slt', 'rl', 'init', 'rrk', 'fs', \
                'iht', 'aht', 'dm', 'lr', 'ws', 'me', 'bs', 'cbs', \
                'ies', 'rtopk']

bert_init_models = {'A':"fold_{}_test_plain_init/model_best.ckpt", \
                    'B':"fold_{}_plain_hist1_fix_bert/model_best.ckpt"}
paras = [
    # train init 0,1 bert
    #(1, '', 'plain_transformer', 'none', 'F', 'T', 'T', 'F', 0, 3, 'noam', 0.001, 800, 5, 2, 4, 1, 10, 'train', 'F', 1, 50, 10),
    # collect cq rank with 0-1 trained bert
    #(1, '', 'plain_transformer', 'none', 'F', 'T', 'T', 'F', 0, 3, 'noam', 0.001, 800, 5, 2, 4, 1, 10, 'test', 'T', 1, 50, 10),
    # train 4,1,0 bert with hist_turn=0; fix bert
    #(1, 'A', 'plain_transformer', 'none', 'F', 'F', 'T', 'T', 0, 1, 'noam', 0.001, 800, 5, 2, 4, 1, 10, 'train', 'F', 1, 5, 5),
    # based on 0,1 trained bert, train selector
    #(1, 'A', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 0, 3, 'noam', 0.001, 800, 5, 2, 4, 1, 50),
    # based on 4,1,0 trained bert, train selector
    #(1, 'B', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'adam', 0.0001, 800, 5, 2, 4, 1, 50),

    (2, 'A', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'adam', 0.0005, 800, 5, 2, 4, 1, 50), #select
    (3, 'A', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'noam', 0.001, 800, 5, 2, 4, 1, 50), #select
    (4, 'A', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'adam', 0.0005, 800, 5, 2, 4, 1, 50), #select
    (5, 'B', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'adam', 0.0005, 800, 5, 2, 4, 1, 50), #select

    #(1, 'A', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'adam', 0.0005, 800, 5, 2, 4, 1, 50),
    #(3, 'A', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'adam', 0.0005, 800, 5, 2, 4, 1, 50),
    #(4, 'A', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'adam', 0.0005, 800, 5, 2, 4, 1, 50), #select

    #(1, 'A', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'adam', 0.0001, 800, 5, 2, 4, 1, 50),
    #(3, 'A', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'adam', 0.0001, 800, 5, 2, 4, 1, 50),
    #(4, 'A', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'adam', 0.0001, 800, 5, 2, 4, 1, 50),

    #(1, 'B', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'adam', 0.0005, 800, 5, 2, 4, 1, 50),
    #(3, 'B', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'adam', 0.0005, 800, 5, 2, 4, 1, 50),
    #(4, 'B', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'adam', 0.0005, 800, 5, 2, 4, 1, 50),

    #(1, 'B', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'adam', 0.0001, 800, 5, 2, 4, 1, 50),
    #(3, 'B', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'adam', 0.0001, 800, 5, 2, 4, 1, 50),
    #(4, 'B', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'adam', 0.0001, 800, 5, 2, 4, 1, 50),
    #(2, 'A', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'noam', 0.0005, 800, 5, 2, 4, 1, 50),
    #(2, 'A', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'noam', 0.001, 500, 5, 2, 4, 1, 50),
    #(2, 'A', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'adam', 0.0005, 800, 5, 2, 4, 1, 50), #select
    #(2, 'A', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'adam', 0.0001, 800, 5, 2, 4, 1, 50),
    #(2, 'A', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'adam', 0.00005, 800, 5, 2, 4, 1, 50),
    # based on 4,1,0 trained bert, train selector
    #(2, 'B', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 0, 3, 'noam', 0.001, 800, 5, 2, 4, 1, 50),
    #(2, 'B', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'noam', 0.001, 800, 5, 2, 4, 1, 50),
    #(2, 'B', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'noam', 0.0005, 800, 5, 2, 4, 1, 50),
    #(2, 'B', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'noam', 0.001, 500, 5, 2, 4, 1, 50),
    #(2, 'B', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'adam', 0.0005, 800, 5, 2, 4, 1, 50),
    #(2, 'B', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'adam', 0.0001, 800, 5, 2, 4, 1, 50),
    #(2, 'B', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'adam', 0.00005, 800, 5, 2, 4, 1, 50),

    #(5, 'A', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'noam', 0.0005, 800, 5, 2, 4, 1, 50),
    #(5, 'A', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'noam', 0.001, 500, 5, 2, 4, 1, 50),
    #(5, 'A', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'adam', 0.0005, 800, 5, 2, 4, 1, 50),
    #(5, 'A', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'adam', 0.0001, 800, 5, 2, 4, 1, 50),
    #(5, 'A', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'adam', 0.00005, 800, 5, 2, 4, 1, 50),
    #(5, 'B', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'noam', 0.0005, 800, 5, 2, 4, 1, 50),
    #(5, 'B', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'noam', 0.001, 500, 5, 2, 4, 1, 50),
    #(5, 'B', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'adam', 0.0005, 800, 5, 2, 4, 1, 50), #select
    #(5, 'B', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'adam', 0.0001, 800, 5, 2, 4, 1, 50),
    #(5, 'B', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'adam', 0.00005, 800, 5, 2, 4, 1, 50),
    #(3, 'A', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 0, 3, 'noam', 0.001, 800, 5, 2, 4, 1, 50),
    #(3, 'A', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'noam', 0.001, 800, 5, 2, 4, 1, 50), # select
    #(3, 'B', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 0, 3, 'noam', 0.001, 800, 5, 2, 4, 1, 50),
    #(3, 'B', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'noam', 0.001, 800, 5, 2, 4, 1, 50),
    #(4, 'A', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 0, 3, 'noam', 0.001, 800, 5, 2, 4, 1, 50),
    #(4, 'A', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'noam', 0.001, 800, 5, 2, 4, 1, 50),
    #(4, 'B', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 0, 3, 'noam', 0.001, 800, 5, 2, 4, 1, 50),
    #(4, 'B', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'noam', 0.001, 800, 5, 2, 4, 1, 50),
    #(5, 'A', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 0, 3, 'noam', 0.001, 800, 5, 2, 4, 1, 50),
    #(5, 'A', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'noam', 0.001, 800, 5, 2, 4, 1, 50),
    #(5, 'B', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 0, 3, 'noam', 0.001, 800, 5, 2, 4, 1, 50),
    #(5, 'B', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'noam', 0.001, 800, 5, 2, 4, 1, 50),
    # based on 0,1 trained bert, train selector with RL
    #(1, 'A', 'plain_transformer', 'plain', 'T', 'F', 'T', 'T', 0, 0, 'noam', 0.001, 800, 5, 2, 4, 1, 10, 'train', 'F', 1, 5, 5),
    #(1, 'A', 'plain_transformer', 'plain', 'T', 'F', 'T', 'T', 0, 0, 'noam', 0.001, 800, 5, 2, 4, 1, 20, 'train', 'F', 1, 5, 5),



    ]

if __name__ == '__main__':
    fscript = open("run_model.sh", 'w')
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default='log_cq_ranker')
    parser.add_argument("--script_dir", type=str, default='script_cq_ranker')
    args = parser.parse_args()
    os.system("mkdir -p %s" % args.log_dir)
    os.system("mkdir -p %s" % args.script_dir)
    job_id = 1
    for para in paras:
        cmd_arr = []
        cmd_arr.append(script_path)
        dataset = "fold_%d" % para[0]
        os.system("mkdir -p {}/{}".format(args.log_dir, dataset))
        cmd_arr.append('--pretrained_bert_path {}'.format(BERT_PRETRAIN))
        cmd_arr.append('--data_dir {}'.format(WORKING_DIR))
        input_train_dir = os.path.join(WORKING_DIR, dataset)
        cmd_arr.append('--input_train_dir {}'.format(input_train_dir))
        init_model = 'n' if para[1] not in bert_init_models \
            else os.path.join(OUTPUT_DIR, bert_init_models[para[1]].format(para[0]))
        cmd_arr.append('--init_model %s' % init_model)
        output_path = "%s/%s" % (OUTPUT_DIR, dataset)
        run_name = "_".join(["{}{}".format(x,y) for x,y in zip(short_names, para)])
        model_name = "_".join(["{}{}".format(x,y) for x,y in zip(short_names[1:], para[1:])])
        save_dir = os.path.join(output_path, model_name)
        cur_cmd_option = " ".join(["--{} {}".format(x,y) for x,y in zip(para_names[2:], para[2:])])#reranktopk
        cmd_arr.append(cur_cmd_option)
        cmd_arr.append("--save_dir %s" % save_dir)
        model_name = "{}_{}".format(dataset, model_name)
        cmd = " " .join(cmd_arr)
        cmd_arr.append("--mode test")
        #cmd_arr.append("--rerank_topk 50")
        test_cmd = []
        for i in range(1, 5):
            test_cmd.append(" ".join(cmd_arr + ["--eval_k %d --rank_cutoff %d --eval_pos %d" % (i,i,i)]))
        #test1_cmd = " ".join(cmd_arr + ["--eval_k 1"])
        #test2_cmd = " ".join(cmd_arr + ["--eval_k 5"])
        #print(cmd)
        #os.system(cmd)
        fname = "%s/%s.sh" % (args.script_dir, run_name)
        with open(fname, 'w') as fout:
            fout.write(config_str)
            fout.write("#SBATCH --job-name=%d.sh\n" % job_id)
            fout.write("#SBATCH --output=%s/%s/%s.txt\n" % (args.log_dir, dataset, model_name))
            fout.write("#SBATCH -e %s/%s/%s.err.txt\n" % (args.log_dir, dataset, model_name))
            fout.write("\n")
            #fout.write("%s\n" % cmd)
            #fout.write("%s\n" % test1_cmd)
            for i in range(len(test_cmd)):
                fout.write("%s\n" % test_cmd[i])
            #fout.write("%s\n" % test2_cmd)
            fout.write("exit\n")

        fscript.write("sbatch %s\n" % fname)
        job_id += 1
    fscript.close()



