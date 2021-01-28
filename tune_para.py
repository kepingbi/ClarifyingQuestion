import sys
import os
import argparse

#titanx-long #1080ti-long #2080ti-long   # Partition to submit to
#SBATCH --mem=96000    # Memory in MB per node allocated
config_str = """#!/bin/bash

#SBATCH --partition=titanx-long
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=64000    # Memory in MB per node allocated
#SBATCH --ntasks-per-node=4
"""

WORKING_DIR="/mnt/nfs/scratch1/kbi/conv_search/working/cq_ranker"
OUTPUT_DIR="/mnt/nfs/scratch1/kbi/conv_search/output/cq_ranker"

script_path = "python main.py"

para_names = ['fold', 'init_model', 'model_name', \
    'selector', 'rl', 'init', 'rerank', 'fix_scorer', 'min_hist_turn', 'max_hist_turn', \
        'decay_method', 'lr', 'warmup_steps', 'max_train_epoch', 'batch_size', 'candi_batch_size', \
        'inter_embed_size', 'rerank_topk', 'mode', 'init_cq', 'eval_k', 'rank_cutoff', 'eval_pos']
short_names = ['im', 'mn', 'slt', 'rl', 'init', 'rrk', 'fs', \
                'iht', 'aht', 'dm', 'lr', 'ws', 'me', 'bs', 'cbs', \
                'ies', 'rtopk', '', '', '', '']

bert_init_models = {'A':"fold_{}_test_plain_init/model_best.ckpt", \
                    'B':"fold_{}_plain_hist1_fix_bert/model_best.ckpt"}
paras = [
    # train init 0,1 bert
    (1, '', 'plain_transformer', 'none', 'F', 'T', 'T', 'F', 0, 3, 'noam', 0.001, 800, 5, 2, 4, 1, 10, 'train', 'F', 1, 50, 10), 
    # collect cq rank with 0-1 trained bert
    (1, '', 'plain_transformer', 'none', 'F', 'T', 'T', 'F', 0, 3, 'noam', 0.001, 800, 5, 2, 4, 1, 10, 'test', 'T', 1, 50, 10), 
    # train 4,1,0 bert with hist_turn=0; fix bert
    (1, '', 'plain_transformer', 'none', 'F', 'F', 'T', 'T', 0, 1, 'noam', 0.001, 800, 5, 2, 4, 1, 10, 'train', 'F', 1, 5, 5), 
    # based on 0,1 trained bert, train selector
    (1, 'A', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 0, 3, 'noam', 0.001, 800, 5, 2, 4, 1, 10, 'train', 'F', 1, 5, 5), 
    (1, 'A', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'noam', 0.001, 800, 5, 2, 4, 1, 10, 'train', 'F', 1, 5, 5), 
    # based on 4,1,0 trained bert, train selector
    (1, 'B', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 0, 3, 'noam', 0.001, 800, 5, 2, 4, 1, 10, 'train', 'F', 1, 5, 5), 
    (1, 'B', 'plain_transformer', 'plain', 'F', 'F', 'T', 'T', 1, 3, 'noam', 0.001, 800, 5, 2, 4, 1, 10, 'train', 'F', 1, 5, 5), 
    # based on 0,1 trained bert, train selector with RL
    (1, 'A', 'plain_transformer', 'plain', 'T', 'F', 'T', 'T', 0, 0, 'noam', 0.001, 800, 5, 2, 4, 1, 10, 'train', 'F', 1, 5, 5), 
    (1, 'A', 'plain_transformer', 'plain', 'T', 'F', 'T', 'T', 0, 0, 'noam', 0.001, 800, 5, 2, 4, 1, 20, 'train', 'F', 1, 5, 5), 


 
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
        input_train_dir = os.path.join(WORKING_DIR, dataset)
        cmd_arr.append('--input_train_dir {}'.format(input_train_dir))
        init_model = 'n' if para[1] not in bert_init_models \
            else os.path.join(OUTPUT_DIR, bert_init_models[para[1]].format(para[0]))
        cmd_arr.append('--init_model %s' % init_model)
        output_path = "%s/%s" % (OUTPUT_DIR, dataset)
        run_name = "_".join(["{}{}".format(x,y) for x,y in zip(short_names, para[1:])])
        model_name = "_".join(["{}{}".format(x,y) for x,y in zip(short_names[:-4], para[1:])])
        save_dir = os.path.join(output_path, model_name)
        cur_cmd_option = " ".join(["--{} {}".format(x,y) for x,y in zip(para_names[2:], para[2:])])
        cmd_arr.append(cur_cmd_option)
        cmd_arr.append("--save_dir %s" % save_dir)
        model_name = "{}_{}".format(dataset, model_name)
        cmd = " " .join(cmd_arr)
        #print(cmd)
        #os.system(cmd)
        fname = "%s/%s.sh" % (args.script_dir, run_name)
        with open(fname, 'w') as fout:
            fout.write(config_str)
            fout.write("#SBATCH --job-name=%d.sh\n" % job_id)
            fout.write("#SBATCH --output=%s/%s/%s.txt\n" % (args.log_dir, dataset, model_name))
            fout.write("#SBATCH -e %s/%s/%s.err.txt\n" % (args.log_dir, dataset, model_name))
            fout.write("\n")
            fout.write(cmd)
            fout.write("\n")
            fout.write("exit\n")

        fscript.write("sbatch %s\n" % fname)
        job_id += 1
    fscript.close()



