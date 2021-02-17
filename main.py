"""
main entry of the script, train, validate and test
"""
import torch
import argparse
import random
import glob
import os

from others.logging import logger, init_logger
from models.cq_ranker import ClarifyQuestionRanker, build_optim
from models.context_ranker import ContextRanker
from models.ref_word_ranker import RefWordsRanker
from data.data_util import GlobalConvSearchData, ConvSearchData
from trainer import Trainer
from rl_trainer import RLTrainer
from data.cq_retriever_dataset import ClarifyQuestionDataset

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=666, type=int)
    parser.add_argument("--train_from", default='')
    parser.add_argument("--init_model", default='')
    parser.add_argument("--init_rankfile", default='')
    parser.add_argument("--rerank_topk", type=int, default=50, help="rerank how many cqs to use during interactions")
    parser.add_argument("--fix_conv_turns", type=int, default=0, \
        help="how many turns are fixed during test time. (0 or 1:enumerate all the cq with label 1 as history")
    parser.add_argument("--model_name", default='plain_transformer',
            choices=['ref_transformer', 'plain_transformer', 'avg_transformer'],
            help="which type of model is used to train")
    parser.add_argument("--selector", default='none',
            choices=['ref', "wref", 'plain', "none"], help="which type of model is used as selector")
    parser.add_argument("--sel_struct", default='concat',
            choices=['concat', 'numeric'], help="which type of model is used as selector")
    parser.add_argument("--rl", type=str2bool, nargs='?',const=True,default=False,
            help="whether to use reinforcement learning to train.")
    parser.add_argument("--punish_sim", type=str2bool, nargs='?',const=True,default=False,
            help="whether to punish grade-1 cq in reinforcement learning.")
    parser.add_argument("--punish_turns", type=str2bool, nargs='?',const=True,default=False,
            help="whether to punish long turns in reinforcement learning.")
    parser.add_argument("--init", type=str2bool, nargs='?',const=True,default=False,
            help="whether to train a model for initial retrieval.")
    parser.add_argument("--init_cq", type=str2bool, nargs='?',const=True,default=False,
            help="whether to collect ranklist for cqs.")
    parser.add_argument("--fix_scorer", type=str2bool, nargs='?',const=True,default=True,
            help="whether to fix the initial model for relevance matching.")
    parser.add_argument("--inter_embed_size", type=int, default=1, help="Size of each embedding after transformers.")
    parser.add_argument("--sep_selector", type=str2bool, nargs='?',const=True,default=True,
            help="whether to use separate model for the selector or share the parameters with retriever.")
    parser.add_argument("--projector_embed_size", type=int, default=32,\
        help="Size of each embedding after transformers for context_ranker.")
    parser.add_argument("--rerank", type=str2bool, nargs='?',const=True,default=True,
            help="whether to rerank results from initial retrieval.")
    parser.add_argument("--rankfname", default="test.best_model.ranklist")
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--tweight", default=0.9, type=float)
    parser.add_argument("--mmr_sim", default="log_sigmoid", type=str, choices=["log_sigmoid", "scores", "sigmoid"])
    parser.add_argument("--aggr", default="max", type=str, choices=["max", "mean"])
    parser.add_argument("--sigmoid_t", default=0.1, type=float)
    parser.add_argument("--sigmoid_cq", default=0.1, type=float)
    parser.add_argument("--pos_cq_thre", default=0.8, type=float, help="")
    parser.add_argument("--hidden_size", default=1, type=int, help="hidden size for ref_word_ranker")
    parser.add_argument("--token_dropout", default=0.1, type=float)
    parser.add_argument("--optim", type=str, default="adam", help="sgd or adam")
    parser.add_argument("--lr", default=0.001, type=float) #0.002
    parser.add_argument("--beta1", default= 0.9, type=float)
    parser.add_argument("--beta2", default=0.999, type=float)
    parser.add_argument("--decay_method", default='noam', choices=['noam', 'adam'],type=str) #warmup learning rate then decay
    parser.add_argument("--warmup_steps", default=800, type=int) #10000
    parser.add_argument("--max_grad_norm", type=float, default=5.0,
                            help="Clip gradients to this norm.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--l2_lambda", type=float, default=0.0,
                            help="The lambda for L2 regularization.")
    parser.add_argument("--batch_size", type=int, default=2,
                            help="Batch size to use during training.")
    parser.add_argument("--has_valid", type=str2bool, nargs='?',const=True,default=True,
            help="whether there is validation set; if not use test as validation.")
    parser.add_argument("--valid_batch_size", type=int, default=4,
                            help="Batch size for validation to use during training.")
#     parser.add_argument("--valid_candi_size", type=int, default=500, #
#                             help="Random products used for validation. When it is 0 or less, all the products are used.")
#     parser.add_argument("--test_candi_size", type=int, default=-1, #
#                             help="When it is 0 or less, all the products are used. Otherwise, test_candi_size samples from ranklist will be reranked")
    parser.add_argument("--candi_batch_size", type=int, default=4,
                            help="Batch size for validation to use during training.")
    parser.add_argument("--num_workers", type=int, default=4,
                            help="Number of processes to load batches of data during training.")
    parser.add_argument("--data_dir", type=str, default="/net/home/kbi/projects/conv_search/qulac/data/working", help="Data directory")
    parser.add_argument("--input_train_dir", type=str, default="/net/home/kbi/projects/conv_search/qulac/data/working/fold_1", help="The directory of training and testing data")
    parser.add_argument("--save_dir", type=str, default="/net/home/kbi/projects/conv_search/qulac/data/output/test/fold_1_test", help="Model directory & output directory")
    parser.add_argument("--log_file", type=str, default="train.log", help="log file name")
    parser.add_argument("--pretrained_bert_path", type=str, default="/mnt/scratch/kbi/conv_search/qulac/data/bert_pretrain/finetuned_lm", help="Embeddings of locally pretrained BERT")
    parser.add_argument("--embedding_size", type=int, default=128, help="Size of each embedding.")
    parser.add_argument("--ff_size", type=int, default=512, help="size of feedforward layers in transformers.")
    parser.add_argument("--heads", default=8, type=int, help="attention heads in transformers")
    parser.add_argument("--inter_layers", default=1, type=int, help="transformer layers")
    parser.add_argument("--max_train_epoch", type=int, default=5,
                            help="Limit on the epochs of training (0: no limit).")
    parser.add_argument("--start_epoch", type=int, default=0,
                            help="the epoch where we start training.")
    parser.add_argument("--steps_per_checkpoint", type=int, default=200,
                            help="How many training steps to do per checkpoint.")
    parser.add_argument("--neg_per_pos", type=int, default=5,
                            help="How many negative samples used to pair with postive results.")
    parser.add_argument("--sparse_emb", action='store_true',
                            help="use sparse embedding or not.")
    parser.add_argument("--scale_grad", action='store_true',
                            help="scale the grad of word and av embeddings.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "valid", "test", "baseline"])
    parser.add_argument("--doc_topk", type=int, default=3,
                            help="The number of documents used for reference.")
    parser.add_argument("--cq_topk", type=int, default=15,
                            help="The number of documents used for reference.")
    parser.add_argument("--words_topk", type=int, default=20,
                            help="The number of documents used for reference.")
    parser.add_argument("--rank_cutoff", type=int, default=5,
                            help="Rank cutoff for output ranklists.")
    parser.add_argument("--eval_k", type=int, default=5,
                            help="Iteration for the clarifying questions.")
    parser.add_argument("--eval_pos", type=int, default=5,
                            help="Evaluation Position for NDCG, Precision, and Recall.")
    parser.add_argument("--max_hist_turn", type=int, default=3,
                            help="Maximum Iteration for the clarifying questions.")
    parser.add_argument("--min_hist_turn", type=int, default=1,
                            help="Minimum Iteration for the clarifying questions.")
    parser.add_argument("--max_episode_len", type=int, default=5,
                            help="Iteration for the clarifying questions.")
    parser.add_argument("--gamma", type=float, default=0.9,
                            help="Gamma for RL future reward accumulation.")

    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'], help="use CUDA or cpu")
    return parser.parse_args()

model_flags = ['embedding_size', 'ff_size', 'heads', 'inter_layers']

def create_model(args, load_path='', partial=False):
    """Create model and initialize or load parameters in session."""
        #global_data and conv_data not used yet
    if args.selector == "none":
        # if args.model_name == "ref_transformer" or args.model_name == "plain_transformer":
        model = ClarifyQuestionRanker(args, args.device)
    elif args.selector == "plain":
        model = ContextRanker(args, args.device)
    else:
        model = RefWordsRanker(args, args.device)

    if os.path.exists(load_path):
    #if load_path != '':
        logger.info('Loading checkpoint from %s' % load_path)
        checkpoint = torch.load(load_path,
                                map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])
        # args.start_epoch = checkpoint['epoch']
        if partial:
            model.cq_bert_ranker.load_cp(checkpoint, strict=False)
            # if args.fix_scorer:
            #     logger.info("fix the initial model with BERT")
            #     # for param in model.cq_bert_ranker.parameters():
            #     for param in model.cq_bert_ranker.bert.parameters(): # wo not fixed
            #         param.requires_grad = False
        else:
            model.load_cp(checkpoint, strict=False)
        if args.fix_scorer:
            logger.info("fix BERT in %s" % (type(model)))
            if type(model) == ClarifyQuestionRanker:
                for param in model.bert.parameters(): # wo not fixed
                    param.requires_grad = False
            elif type(model) == ContextRanker:
                for param in model.cq_bert_ranker.bert.parameters(): # wo not fixed
                # for param in model.cq_bert_ranker.parameters(): # wo fixed
                    param.requires_grad = False

        optim = build_optim(args, model, None)
        # optim = build_optim(args, model, checkpoint)
    else:
        logger.info('No available model to load. Build new model.')
        optim = build_optim(args, model, None)
    # logger.info(model)
    return model, optim

def train(args):
    args.start_epoch = 0
    logger.info('Device %s' % args.device)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)

    global_data = GlobalConvSearchData(args, args.data_dir)
    if os.path.exists(args.init_model):
        partial = False if args.selector == "none" else True
        model, optim = create_model(args, args.init_model, partial=partial)
    else:
        model, optim = create_model(args, args.train_from)
    if args.rl:
        trainer = RLTrainer(args, model, optim)
    else:
        trainer = Trainer(args, model, optim)
    train_conv_data = ConvSearchData(args, args.input_train_dir, "train", global_data)
    valid_conv_data = ConvSearchData(args, args.input_train_dir, "valid", global_data)
    best_checkpoint_path = trainer.train(trainer.args, global_data, train_conv_data, valid_conv_data)
    test_conv_data = ConvSearchData(args, args.input_train_dir, "test", global_data)
    best_model, _ = create_model(args, best_checkpoint_path)
    del trainer
    torch.cuda.empty_cache()
    trainer = Trainer(args, best_model, None)
    trainer.test(args, global_data, test_conv_data, args.rankfname)

def validate(args):
    cp_files = sorted(glob.glob(os.path.join(args.save_dir, 'model_epoch_*.ckpt')))
    global_data = GlobalConvSearchData(args, args.data_dir)
    valid_conv_data = ConvSearchData(args, args.input_train_dir, "valid", global_data)
#     valid_dataset = ClarifyQuestionDataset(args, global_data, valid_conv_data)
    best_criterion, best_model = 0, None
    for cur_model_file in cp_files:
        #logger.info("Loading {}".format(cur_model_file))
        cur_model, _ = create_model(args, cur_model_file)
        trainer = Trainer(args, cur_model, None)
        ndcg, prec, recall, mrr = trainer.validate(args, global_data, valid_conv_data)
        logger.info("NDCG@{}:{} P@{}:{} R@{}:{} MRR:{} Model:{}".format(
            args.eval_pos, ndcg, args.eval_pos, prec, args.eval_pos, recall, mrr, cur_model_file))
        criterion = recall if args.init else mrr
        if criterion > best_criterion:
            best_criterion = criterion
            best_model = cur_model_file

    test_conv_data = ConvSearchData(args, args.input_train_dir, "test", global_data)

    logger.info("Best Model: {}".format(best_model))
    best_model, _ = create_model(args, best_model)
    trainer = Trainer(args, best_model, None)
    trainer.test(args, global_data, test_conv_data, args.rankfname)

def test(args):
    global_data = GlobalConvSearchData(args, args.data_dir)
    test_conv_data = ConvSearchData(args, args.input_train_dir, "test", global_data)
    model_path = os.path.join(args.save_dir, 'model_best.ckpt')
    if os.path.exists(args.train_from):
        best_model, _ = create_model(args, args.train_from)
    else:
        best_model, _ = create_model(args, model_path)
    trainer = Trainer(args, best_model, None)
    trainer.test(args, global_data, test_conv_data, args.rankfname)

def main(args):
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    init_logger(os.path.join(args.save_dir, args.log_file))
    logger.info(args)
    if args.mode == "train":
        train(args)
    elif args.mode == "valid":
        validate(args)
    elif args.mode == "test":
        test(args)
    elif args.mode == "baseline":
        global_data = GlobalConvSearchData(args, args.data_dir)
        test_conv_data = ConvSearchData(args, args.input_train_dir, "test", global_data)
        trainer = Trainer(args, None, None)
        trainer.run_baseline(args, global_data, test_conv_data, args.rankfname)
if __name__ == '__main__':
    main(parse_args())
