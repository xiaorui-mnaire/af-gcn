"""
Created on Mar 1, 2022
Pytorch Implementation of AF-GCN in
XiaoRui et al. AF-GCN: Attribute-fusing Graph Convolution Network for Recommendation

Define models here
"""
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go AFGCN")
    parser.add_argument('--bpr_batch', type=int, default=2048,  # 2048
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int, default=64,  # 64
                        help="the embedding size of AFGCN")
    parser.add_argument('--layer', type=int, default=3,  # 3
                        help="the layer num of AFGCN")
    parser.add_argument('--lr', type=float, default=0.001,  # 0.001
                        help="the learning rate")
    parser.add_argument('--decay', type=float, default=1e-4,  # 1e-4
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int, default=0,  # 0
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float, default=0.6,  # 0.6
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int, default=100,  # 100
                        help="the fold num used to split large adj matrix, like Movielens 100K")
    parser.add_argument('--testbatch', type=int, default=100,  # 100
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str, default='DoubanBook',
                        help="available datasets: [DoubanBook, Movielens 100K, Movielens 1M]")
    parser.add_argument('--path', type=str, default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?', default="[20]",  # [20]
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int, default=1,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str, default="AFGCN")
    parser.add_argument('--load', type=int, default=0)  # 0
    parser.add_argument('--epochs', type=int, default=1000)  # 1000
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')  # 0
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')  # 0
    parser.add_argument('--seed', type=int, default=2020, help='random seed')  # 2020
    parser.add_argument('--model', type=str, default='afgcn', help='rec-model, support [afgcn]')
    return parser.parse_args()
