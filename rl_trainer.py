from tqdm import tqdm
from others.logging import logger
from data.cq_retriever_dataset import ClarifyQuestionDataset
from data.multi_turn_dataset import MultiTurnDataset
# from data.cq_retriever_dataloader import ClarifyQuestionDataloader
import shutil
import torch
import numpy as np
import random
import os
import time
import sys
from collections import defaultdict, namedtuple
from trainer import Trainer
# from others.evaluate import calc_ndcg, calc_mrr

def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params

ActionReward = namedtuple('ActionReward',
                        ('action', 'prob', 'reward'))

class RLTrainer(Trainer):
    """
    Class that controls the training process.
    """
    def __init__(self, args, model, optim, grad_accum_count=1, 
                n_gpu=1, gpu_rank=1, report_manager=None):
        super(RLTrainer, self).__init__(args, model, optim, grad_accum_count, 
                n_gpu, gpu_rank, report_manager)
        # Basic attributes.

    def train(self, args, global_data, train_conv_data, valid_conv_data):
        """ Train the model with reinforce. 
        """
        logger.info('Start training...')
        train_conv_data.reset_set_name("TrainInference")
        model_dir = args.save_dir
        step_time, loss = 0.,0.
        get_batch_time = 0.0
        start_time = time.time()
        current_step = 0
        best_criterion = 0.
        best_checkpoint_path = ''
        self.model.zero_grad()
        tf_ids = list(train_conv_data.candidate_cq_dic.keys())
        for current_epoch in range(args.start_epoch+1, args.max_train_epoch+1):
            self.model.train()
            logger.info("Initialize epoch:%d" % current_epoch)
            # train_conv_data.initialize_epoch()
            # construct dataset for each query since a single batch can not process more than one query
            random.shuffle(tf_ids)
            with tqdm(total=len(tf_ids), desc=f"Epoch {current_epoch}") as pbar:
                #TODO: random shuffle the keys of topic_facet_id for each epoch
                for topic_facet_id in tf_ids:
                    pbar.update(1)
                    # if topic_facet_id != '102-1':
                    #     continue
                    time_flag = time.time()
                    tf_episode = self.sample_an_episode(args, topic_facet_id, global_data, train_conv_data)
                    #update models
                    if len(tf_episode) == 0:
                        continue
                    get_batch_time += time.time() - time_flag
                    time_flag = time.time()
                    step_loss = self.calc_policy_loss(tf_episode)

                    step_loss.backward()
                    if (current_step + 1) % args.gradient_accumulation_steps == 0:
                        step_loss /= args.gradient_accumulation_steps
                        self.optim.step()
                        self.optim.optimizer.zero_grad()
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
            # ndcg, prec, mrr = 0., 0., 0.
            logger.info("Epoch {}: NDCG@{}:{} P@{}:{} R@{}:{} MRR:{}".format(
                        current_epoch, self.eval_pos, ndcg, self.eval_pos, prec, self.eval_pos, recall, mrr))
            criterion = recall if self.args.init else mrr
            if criterion > best_criterion:
                best_criterion = criterion
                best_checkpoint_path = os.path.join(model_dir, 'model_best.ckpt')
                logger.info("Copying %s to checkpoint %s" % (checkpoint_path, best_checkpoint_path))
                shutil.copyfile(checkpoint_path, best_checkpoint_path)

        return best_checkpoint_path

    def calc_policy_loss(self, tf_episode):
        steps = len(tf_episode)
        discounted_rewards = [x.reward for x in tf_episode]
        log_probs = [x.prob for x in tf_episode]
        # print("rewards", discounted_rewards)
        for t in reversed(range(steps - 1)):
            discounted_rewards[t] += discounted_rewards[t+1] * self.args.gamma
        
        discounted_rewards = np.asarray(discounted_rewards)
        # discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards
        # do not use this to stablize training
        # if discounted_rewards is torch.tensor. torch.tensor([3]).std() will be nan
        policy_gradient = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)
        # TODO: the log_prob is not at the same scale when the candidate sets have different sizes
        loss = torch.stack(policy_gradient).sum()
        # print("probs", torch.stack(log_probs).cpu().numpy().tolist())
        # print("rewards", discounted_rewards, "loss", loss.item())
        return loss
        # policy_gradient.backward()
        # policy_network.optimizer.zero_grad()
        # policy_network.optimizer.step()

    def sample_an_episode(self, args, topic_facet_id, global_data, conv_data, other_pos=2, k_candi=10):
        # for each round, prepare dataset and dataloader
        # get candidate scores
        # output ranked list; freezing hist cq + curent ranked list
        # a single topic as the key
        if len(conv_data.candidate_cq_dic[topic_facet_id]) == 0:
            # conv_data.candidate_cq_dic[topic_facet_id] can be empty
            return []
        tf_top_cq_dic = dict()
        tf_candi_cq_dic = {topic_facet_id: conv_data.candidate_cq_dic[topic_facet_id].copy()}
        cur_pos_set, other_pos_set = conv_data.pos_cq_dic[topic_facet_id]
        # origin_candiset = conv_data.candidate_cq_dic[topic_facet_id]
        # if len(origin_candiset) < args.max_episode_len:
        #     other_pos = args.max_episode_len # make the candidate size not too small
        # new_candi_set = set()
        # if len(cur_pos_set) > 0:
        #     new_candi_set.add(random.choice(list(cur_pos_set)))
        # if len(other_pos_set) > 0:
        #     sample_size = min(len(other_pos_set), other_pos)
        #     new_candi_set = new_candi_set.union(random.sample(other_pos_set, sample_size))
        # origin_candiset = origin_candiset.difference(new_candi_set)
        # if len(origin_candiset) > 0:
        #     sample_size = min(len(origin_candiset), k_candi-other_pos)
        #     new_candi_set = new_candi_set.union(random.sample(origin_candiset, sample_size))
        # if len(new_candi_set) == 0:
        #     return []
        # tf_candi_cq_dic = {topic_facet_id: new_candi_set}
        tf_episode = []
        # print(tf_candi_cq_dic)
        for _ in range(args.max_episode_len):
            test_dataset = self.ExpDataset(
                args, global_data, conv_data, hist_cq_dic=tf_top_cq_dic, candi_cq_dic=tf_candi_cq_dic)
            dataloader = self.ExpDataloader(
                    args, test_dataset, batch_size=args.valid_batch_size, #batch_size
                    shuffle=False, num_workers=args.num_workers)
            if len(test_dataset) == 0: # no enough candiates for ranking
                break
            sorted_tf_cq_scores = self.get_cq_scores_for_tf(args, dataloader)
            # model still be train(); using train mode to get scores so that the loss can be back-propagated
            tfid = list(sorted_tf_cq_scores.keys())[0]
            assert tfid == topic_facet_id
            
            # cq_probs = torch.tensor([y for x,y in sorted_tf_cq_scores[tfid]])
            cq_probs = torch.stack([y for x,y in sorted_tf_cq_scores[tfid]])
            cq_probs = torch.softmax(cq_probs, dim=-1) # only 1 dimension
            # print(cq_probs)
            cq_id = torch.multinomial(cq_probs, 1)
            # cq_id = np.random.choice(len(cq_probs), p=cq_probs.cpu().detach().numpy())
            cq, score = sorted_tf_cq_scores[tfid][cq_id] # id, score
            if tfid not in tf_top_cq_dic:
                tf_top_cq_dic[tfid] = []
            if cq in cur_pos_set:
                reward = 4.
            elif cq in other_pos_set:
                reward = -1.
                # reward = 0.
                hist_cqs = [x for x,y in tf_top_cq_dic[tfid]]
                # reward = - MultiTurnDataset.cq_similarity(
                #     global_data.cq_top_cq_info_dic, hist_cqs, cq)
            else:
                reward = -2.
            # delta NDCG can be used as reward
            tf_top_cq_dic[tfid].append((cq, score))
            tf_episode.append(ActionReward(cq, torch.log(cq_probs[cq_id]), reward))
            if cq in cur_pos_set:
                # print(topic_facet_id, cq)
                break
        return tf_episode

    def get_cq_scores_for_tf(self, args, dataloader):
        # pbar = tqdm(dataloader)
        # pbar.set_description(description)
        all_tf_cq_scores = defaultdict(list)
        for batch_data in dataloader:
        # for batch_data in pbar:
            batch_data = batch_data.to(args.device)
            # print(batch_data.topic_facet_ids)
            # print(batch_data.candi_cq_ids)
            # print(batch_data.hist_cq_ids)
            batch_scores, _ = self.model.test(batch_data)
            #batch_size, candidate_batch_size
            for i in range(len(batch_data.topic_facet_ids)):
                tfid = batch_data.topic_facet_ids[i]
                cur_entry = list(zip(batch_data.candi_cq_ids[i], batch_scores[i]))
                # print(cur_entry)
                all_tf_cq_scores[tfid].extend(cur_entry)
        return all_tf_cq_scores
