from tqdm import tqdm
from others.logging import logger
from data.cq_retriever_dataset import ClarifyQuestionDataset
from data.cq_retriever_dataloader import ClarifyQuestionDataloader
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
        if (model):
            n_params = _tally_parameters(model)
            logger.info('* number of parameters: %d' % n_params)
        #self.device = "cpu" if self.n_gpu == 0 else "cuda"
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager

        self.ExpDataset = ClarifyQuestionDataset
        self.ExpDataloader = ClarifyQuestionDataloader

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
        best_ndcg = 0.
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
            time_flag = time.time()
            for step, batch_data in enumerate(pbar):
                batch_data = batch_data.to(args.device)
                get_batch_time += time.time() - time_flag
                time_flag = time.time()
                step_loss = self.model(batch_data)
                if args.gradient_accumulation_steps > 1:
                    step_loss = step_loss / args.gradient_accumulation_steps
                step_loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
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
            ndcg, prec, mrr = self.validate(args, global_data, valid_conv_data)
            logger.info("Epoch {}: NDCG@5:{} P@5:{} MRR:{}".format(current_epoch, ndcg, prec, mrr))
            if ndcg > best_ndcg:
                best_ndcg = ndcg
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
        ndcg, prec, mrr = self.calc_metrics(sorted_tf_cq_scores, valid_conv_data.pos_cq_dic, args.rank_cutoff)
        return ndcg, prec, mrr

    def test(self, args, global_data, test_conv_data, rankfname="test.best_model.ranklist", cutoff=100):
        k = args.eval_k
        sorted_tf_cq_scores = self.infer_k_iter(args, global_data, test_conv_data, "Test", k, cutoff)
        self.calc_metrics(sorted_tf_cq_scores, test_conv_data.pos_cq_dic, cutoff)
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
        ndcg, mrr, prec = 0, 0, 0
        eval_pos = 5
        for tfid in sorted_tf_cq_scores: # queries with no 
            cur_pos_dic, other_pos_dic = pos_cq_dic[tfid]
            ranklist = [2 if x in cur_pos_dic \
                else 1 if x in other_pos_dic else 0 for x, score in sorted_tf_cq_scores[tfid]]
            iranklist = [2] * len(cur_pos_dic) + [1] * len(other_pos_dic)
            cur_ndcg = calc_ndcg(ranklist, iranklist, pos=eval_pos)
            cur_mrr = calc_mrr(ranklist)
            ndcg += cur_ndcg
            mrr += cur_mrr
            prec += sum([1 if x > 0 else 0 for x in ranklist[:eval_pos]]) / eval_pos
        eval_count = len(sorted_tf_cq_scores)
        ndcg /= eval_count
        mrr /= eval_count
        prec /= eval_count
        logger.info(
            "EvalCount:{} NDCG@5:{} P@5:{} MRR:{}".format(eval_count, ndcg, prec, mrr))
        # some topic_facet_id could have no cur_pos_cq (with label 2), 
        # but they still have other_pos_cq with label 1.
        return ndcg, prec, mrr

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

            end = cutoff if i == k-1 else 1
            for tfid in sorted_tf_cq_scores:
                cur_pos_dic, _ = conv_data.pos_cq_dic[tfid]
                if sorted_tf_cq_scores[tfid][0] in cur_pos_dic:
                    tf_candi_cq_dic.pop(tfid, None)
                if tfid not in tf_top_cq_dic:
                    tf_top_cq_dic[tfid] = []
                tf_top_cq_dic[tfid].extend(sorted_tf_cq_scores[tfid][:end])
        return tf_top_cq_dic