from tqdm import tqdm
from others.logging import logger
from data.cq_retriever_dataset import ClarifyQuestionDataset
from data.cq_retriever_dataloader import ClarifyQuestionDataloader
from data.init_retrieval_dataset import InitRetrievalDataset
from data.init_retrieval_dataloader import InitRetrievalDataloader
from data.multi_turn_dataset import MultiTurnDataset
from data.multi_turn_dataloader import MultiTurnDataloader
from data.ref_word_dataset import RefWordsDataset
from data.ref_word_dataloader import RefWordsDataloader
import shutil
import torch
import numpy as np
import os
import time
import sys
from collections import defaultdict
from others.evaluate import calc_ndcg, calc_mrr

def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params

class Trainer(object):
    """
    Class that controls the training process.
    """
    def __init__(self, args, model, optim, grad_accum_count=1, 
                n_gpu=1, gpu_rank=1, report_manager=None):
        # Basic attributes.
        self.args = args
        self.model = model
        self.optim = optim
        self.eval_pos = args.eval_pos
        if (model):
            n_params = _tally_parameters(model)
            logger.info('* number of parameters: %d' % n_params)
        #self.device = "cpu" if self.n_gpu == 0 else "cuda"
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager
        if self.args.init:
            self.ExpDataloader = InitRetrievalDataloader
            self.ExpDataset = InitRetrievalDataset
        else:
            if self.args.selector == "none":
                self.ExpDataset = ClarifyQuestionDataset
                self.ExpDataloader = ClarifyQuestionDataloader
            if self.args.selector == "plain":
                self.ExpDataset = MultiTurnDataset
                self.ExpDataloader = MultiTurnDataloader
            else:
                self.ExpDataset = RefWordsDataset
                self.ExpDataloader = RefWordsDataloader

    def train(self, args, global_data, train_conv_data, valid_conv_data):
        """
        The main training loops.
        """
        logger.info('Start training...')
        # Set model in training mode.
        model_dir = args.save_dir
        # valid_dataset = self.ExpDataset(args, global_data, valid_conv_data)
        step_time, loss = 0.,0.
        get_batch_time = 0.0
        start_time = time.time()
        current_step = 0
        best_criterion = 0.
        best_checkpoint_path = ''
        for current_epoch in range(args.start_epoch+1, args.max_train_epoch+1):
            self.model.train()
            logger.info("Initialize epoch:%d" % current_epoch)
            train_conv_data.initialize_epoch()
            dataset = self.ExpDataset(args, global_data, train_conv_data)
            dataloader = self.ExpDataloader(
                    args, dataset, batch_size=args.batch_size,
                    shuffle=True, num_workers=args.num_workers)
            pbar = tqdm(dataloader)
            pbar.set_description("[Epoch {}]".format(current_epoch))
            for batch_data in pbar:
                time_flag = time.time()
                batch_data = batch_data.to(args.device)
                get_batch_time += time.time() - time_flag
                time_flag = time.time()
                step_loss = self.model(batch_data)
                step_loss.backward()
                if (current_step + 1) % args.gradient_accumulation_steps == 0:
                    step_loss /= args.gradient_accumulation_steps
                    self.optim.step()
                    self.optim.optimizer.zero_grad()
                    # self.model.zero_grad()

                step_loss = step_loss.item()
                pbar.set_postfix(step_loss=step_loss, lr=self.optim.learning_rate)
                loss += step_loss * args.gradient_accumulation_steps / args.steps_per_checkpoint #convert an tensor with dim 0 to value
                current_step += 1
                step_time += time.time() - time_flag

                # Once in a while, we print statistics.
                if current_step % args.steps_per_checkpoint == 0:
                    logger.info("Epoch %d lr = %5.8f loss = %6.2f time %.2f prepare_time %.2f step_time %.2f" %
                            (current_epoch, self.optim.learning_rate, loss,
                                time.time()-start_time, get_batch_time, step_time))#, end=""
                    step_time, get_batch_time, loss = 0., 0.,0.
                    sys.stdout.flush()
                    start_time = time.time()
            checkpoint_path = os.path.join(model_dir, 'model_epoch_%d.ckpt' % current_epoch)
            self._save(current_epoch, checkpoint_path)
            ndcg, prec, recall, mrr = self.validate(args, global_data, valid_conv_data)

            logger.info("Epoch {}: NDCG@{}:{} P@{}:{} R@{}:{} MRR:{}".format(
                        current_epoch, self.eval_pos, ndcg, self.eval_pos, prec, self.eval_pos, recall, mrr))
            criterion = recall if self.args.init else mrr
            if criterion > best_criterion:
                best_criterion = criterion
                best_checkpoint_path = os.path.join(model_dir, 'model_best.ckpt')
                logger.info("Copying %s to checkpoint %s" % (checkpoint_path, best_checkpoint_path))
                shutil.copyfile(checkpoint_path, best_checkpoint_path)
        return best_checkpoint_path

    def _save(self, epoch, checkpoint_path):
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'opt': self.args,
            'optim': self.optim,
        }
        #model_dir = "%s/model" % (self.args.save_dir)
        #checkpoint_path = os.path.join(model_dir, 'model_epoch_%d.ckpt' % epoch)
        logger.info("Saving checkpoint %s" % checkpoint_path)
        torch.save(checkpoint, checkpoint_path)

    def validate(self, args, global_data, valid_conv_data):
        """ Validate model.
        """
        k = args.eval_k
        sorted_tf_cq_scores = self.infer_k_iter(args, global_data, valid_conv_data, "Validation", k, args.rank_cutoff)
        ndcg, prec, recall, mrr = self.calc_metrics(sorted_tf_cq_scores, valid_conv_data.pos_cq_dic, args.rank_cutoff)
        return ndcg, prec, recall, mrr

    def test(self, args, global_data, test_conv_data, rankfname="test.best_model.ranklist"):
        k = args.eval_k
        cutoff = args.rank_cutoff
        sorted_tf_cq_scores = self.infer_k_iter(args, global_data, test_conv_data, "Test", k, cutoff)
        self.calc_metrics(sorted_tf_cq_scores, test_conv_data.pos_cq_dic, cutoff)
        rankfname = "%s.%diter.len%d" % (rankfname, k, cutoff)
        if args.init_cq:
            rankfname += ".cq"
        if "rerank" in args.init_rankfile:
            rankfname += ".ql"
        if args.rerank:
            rankfname += ".rerank"
        output_path = os.path.join(args.save_dir, rankfname)
        self.write_ranklist(output_path, sorted_tf_cq_scores, cutoff)

    def write_ranklist(self, output_path, sorted_tf_cq_scores, cutoff, model_name="CQRanker"):
        with open(output_path, 'w') as rank_fout:
            for tfid in sorted(sorted_tf_cq_scores.keys()):
                sorted_tf_cq_scores[tfid] = sorted_tf_cq_scores[tfid][:cutoff]
                # cq_id_ranklist = [x for x,y in sorted_tf_cq_scores[tfid]]
                # cq_id_ranklist = cq_id_ranklist[:cutoff]
                for rank in range(len(sorted_tf_cq_scores[tfid])):
                    cq_id, score = sorted_tf_cq_scores[tfid][rank]
                    line = "%s Q0 %s %d %f %s\n" \
                            % (tfid, cq_id, rank+1, score, model_name)
                    rank_fout.write(line)

    def calc_metrics(self, sorted_tf_cq_scores, pos_cq_dic, cutoff=100):
        ndcg, mrr, prec, recall = 0, 0, 0, 0
        eval_pos = self.eval_pos
        for tfid in sorted_tf_cq_scores: # queries with no 
            cur_pos_dic, other_pos_dic = pos_cq_dic.get(tfid, [[],[]])
            ranklist = [2 if x in cur_pos_dic \
                else 1 if x in other_pos_dic else 0 for x, score in sorted_tf_cq_scores[tfid]]
            iranklist = [2] * len(cur_pos_dic) + [1] * len(other_pos_dic)
            iranklist = iranklist[:len(ranklist)] # make them the same length
            cur_ndcg = calc_ndcg(ranklist, iranklist, pos=eval_pos)
            cur_mrr = calc_mrr(ranklist)
            ndcg += cur_ndcg
            mrr += cur_mrr
            prec += sum([1 if x > 0 else 0 for x in ranklist[:eval_pos]]) \
                / min(len(ranklist), eval_pos)
            recall += sum([1 if x > 0 else 0 for x in ranklist[:eval_pos]]) \
                / (max(1, len(cur_pos_dic)+len(other_pos_dic)))
        eval_count = len(sorted_tf_cq_scores)
        ndcg /= eval_count
        mrr /= eval_count
        prec /= eval_count
        recall /= eval_count
        logger.info(
            "EvalCount:{} NDCG@{}:{} P@{}:{} R@{}:{} MRR:{}".format(
                eval_count, eval_pos, ndcg, eval_pos, prec, eval_pos, recall, mrr))
        # some topic_facet_id could have no cur_pos_cq (with label 2), 
        # but they still have other_pos_cq with label 1.
        return ndcg, prec, recall, mrr

    def get_entry_scores(self, args, dataloader, description):
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(dataloader)
            pbar.set_description(description)
            all_tf_cq_scores = defaultdict(list)
            for batch_data in pbar:
                batch_data = batch_data.to(args.device)
                batch_scores, _ = self.model.test(batch_data)
                #batch_size, candidate_batch_size
                batch_scores = batch_scores.cpu().numpy().tolist()
                for i in range(len(batch_data.topic_facet_ids)):
                    tfid = batch_data.topic_facet_ids[i]
                    cur_entry = list(zip(batch_data.candi_cq_ids[i], batch_scores[i]))
                    # print(cur_entry)
                    all_tf_cq_scores[tfid].extend(cur_entry)
        for tfid in all_tf_cq_scores:
            # print(all_tf_cq_scores[tfid])
            all_tf_cq_scores[tfid].sort(key=lambda x:x[1], reverse=True)
        # print(len(all_tf_cq_scores))
        return all_tf_cq_scores

    def infer_k_iter(self, args, global_data, conv_data, description, k=3, cutoff=100):
        # for each round, prepare dataset and dataloader
        # get candidate scores
        # output ranked list; freezing hist cq + curent ranked list
        # tf_top_cq_dic = defaultdict(list)
        tf_top_cq_dic = dict()
        tf_candi_cq_dic = conv_data.candidate_cq_dic.copy()
        for i in range(k):
            test_dataset = self.ExpDataset(
                args, global_data, conv_data, hist_cq_dic=tf_top_cq_dic, candi_cq_dic=tf_candi_cq_dic)
            dataloader = self.ExpDataloader(
                    args, test_dataset, batch_size=args.valid_batch_size, #batch_size
                    shuffle=False, num_workers=args.num_workers)
            sorted_tf_cq_scores = self.get_entry_scores(args, dataloader, description)
            # print(len(sorted_tf_cq_scores), len(tf_candi_cq_dic))
            for tfid in sorted_tf_cq_scores:
                cur_pos_dic, _ = conv_data.pos_cq_dic.get(tfid, [[],[]])
                if sorted_tf_cq_scores[tfid][0][0] in cur_pos_dic:
                    # cq_id, score # tf id of the top 1 cq
                    tf_candi_cq_dic.pop(tfid, None)
                if tfid not in tf_top_cq_dic:
                    tf_top_cq_dic[tfid] = []
                end = cutoff - len(tf_top_cq_dic[tfid]) if i == k-1 else 1
                tf_top_cq_dic[tfid].extend(sorted_tf_cq_scores[tfid][:end])
        for tfid in tf_top_cq_dic:
            cur_pos_dic, _ = conv_data.pos_cq_dic.get(tfid, [[],[]])
            # print(tf_top_cq_dic[tfid])
            for rank in range(len(tf_top_cq_dic[tfid])):
                # print(rank, len(tf_top_cq_dic[tfid]), tf_top_cq_dic[tfid][rank])
                if k > 1 and tf_top_cq_dic[tfid][rank][0] in cur_pos_dic:
                    # iterative interaction, when label-2 cq is find, stop iterating. 
                    tf_top_cq_dic[tfid] = tf_top_cq_dic[tfid][:rank+1]
                    break
        return tf_top_cq_dic

    def baseline_k_iter(self, args, global_data, conv_data, description, k=3, cutoff=100):
        # for each round, prepare dataset and dataloader
        # get candidate scores
        # output ranked list; freezing hist cq + curent ranked list
        # tf_top_cq_dic = defaultdict(list)
        sigmoid = torch.nn.Sigmoid()
        tf_top_cq_dic = dict()
        tf_candi_cq_dic = conv_data.candidate_cq_dic.copy() # use candidates for all the tfids
        candi_sim_wrt_tq_dic = global_data.collect_candidate_sim_wrt_tq(
            tf_candi_cq_dic, global_data.cq_cq_rank_dic)
        for i in range(k):
            sorted_tf_cq_scores = dict()
            print(len(tf_candi_cq_dic))
            for tfid in tf_candi_cq_dic:
                tid, fid = tfid.split('-')
                candi_scores = []
                candi_cqs = []
                hist_cq_set = set([cq for cq,score in tf_top_cq_dic.get(tfid, [])])
                candi_cq_set = set([cq for cq, score in tf_candi_cq_dic[tfid]]).difference(hist_cq_set)
                for candi_cq in candi_cq_set:
                    candi_cqs.append(candi_cq)
                    scores = []
                    if "%s-X" % tid not in candi_sim_wrt_tq_dic[candi_cq][tid]:
                        sim_wrt_t = 0.
                        print(candi_cq, tid)
                        print(tf_candi_cq_dic[tfid])
                    else:
                        sim_wrt_t = candi_sim_wrt_tq_dic[candi_cq][tid]["%s-X" % tid]
                    scores.append(sim_wrt_t)
                    if tfid in tf_top_cq_dic:
                        for hist_cq, _ in tf_top_cq_dic[tfid]:
                            if hist_cq not in candi_sim_wrt_tq_dic[candi_cq][tid]:
                                sim_wrt_hist = 0.
                            else:
                                sim_wrt_hist = candi_sim_wrt_tq_dic[candi_cq][tid][hist_cq]
                            scores.append(sim_wrt_hist)
                    candi_scores.append(scores)
                candi_scores = torch.tensor(candi_scores).to(args.device)
                t_sim = torch.log(sigmoid(candi_scores[:,0] * args.sigmoid_t))
                hist_sim = torch.log(1 - sigmoid(candi_scores[:, 1:] * args.sigmoid_cq))
                # print(tfid, t_sim)
                # print(tfid, tf_top_cq_dic.get(tfid, None), hist_sim)
                if hist_sim.size(-1) > 0:
                    sim = t_sim * args.tweight + hist_sim.mean(dim=-1) * (1 - args.tweight)
                else:
                    sim = t_sim
                sorted_scores = list(zip(candi_cqs, sim.cpu().tolist()))
                sorted_scores.sort(key=lambda x:x[1], reverse=True)
                sorted_tf_cq_scores[tfid] = sorted_scores
                # print(sorted_scores)
            for tfid in sorted_tf_cq_scores:
                cur_pos_dic, _ = conv_data.pos_cq_dic.get(tfid, [[],[]])
                if sorted_tf_cq_scores[tfid][0][0] in cur_pos_dic:
                    # cq_id, score # tf id of the top 1 cq
                    tf_candi_cq_dic.pop(tfid, None)
                if tfid not in tf_top_cq_dic:
                    tf_top_cq_dic[tfid] = []
                end = cutoff - len(tf_top_cq_dic[tfid]) if i == k-1 else 1
                tf_top_cq_dic[tfid].extend(sorted_tf_cq_scores[tfid][:end])
        for tfid in tf_top_cq_dic:
            cur_pos_dic, _ = conv_data.pos_cq_dic.get(tfid, [[],[]])
            # print(tf_top_cq_dic[tfid])
            for rank in range(len(tf_top_cq_dic[tfid])):
                # print(rank, len(tf_top_cq_dic[tfid]), tf_top_cq_dic[tfid][rank])
                # if k > 1 and tf_top_cq_dic[tfid][rank][0] in cur_pos_dic:
                if tf_top_cq_dic[tfid][rank][0] in cur_pos_dic:
                    # iterative interaction, when label-2 cq is find, stop iterating. 
                    tf_top_cq_dic[tfid] = tf_top_cq_dic[tfid][:rank+1]
                    break
        return tf_top_cq_dic

    def run_baseline(self, args, global_data, test_conv_data, rankfname="test.best_model.ranklist"):
        k = args.eval_k
        cutoff = args.rank_cutoff
        sorted_tf_cq_scores = self.baseline_k_iter(args, global_data, test_conv_data, "Test", k, cutoff)
        self.calc_metrics(sorted_tf_cq_scores, test_conv_data.pos_cq_dic, cutoff)
        self.eval_pos = 1
        self.calc_metrics(sorted_tf_cq_scores, test_conv_data.pos_cq_dic, cutoff)
        self.eval_pos = 2
        self.calc_metrics(sorted_tf_cq_scores, test_conv_data.pos_cq_dic, cutoff)
        rankfname = "baseline.%s.%diter.len%d" % (rankfname, k, cutoff)
        output_path = os.path.join(args.save_dir, rankfname)
        self.write_ranklist(output_path, sorted_tf_cq_scores, cutoff)
