# -*- coding: utf-8 -*-

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default= 21,
                    help='seed 777')
parser.add_argument('--batch_size', type=int, default= 48,
                    help='batch size 128')
parser.add_argument('--lr', type=float, default= 0.0005,
                    help='learning rate 0.0005')
parser.add_argument('--weight_decay', type=float, default= 0.001,
                    help='weight decay 0.001')
parser.add_argument('--nhid', type=int, default= 128,
                    help='hidden size 128')
parser.add_argument('--pooling_ratio', type=float, default=0.8,
                    help='pooling ratio 0.7')
parser.add_argument('--dropout_ratio', type=float, default=0.45,
                    help='dropout ratio')
parser.add_argument('--epochs', type=int, default=10,
                    help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=50,
                    help='patience for earlystopping')
parser.add_argument('--pooling_layer_type', type=str, default='GCNConv',
                    help='....')
parser.add_argument('--milestones', type=list, default=[100],
                    help='number .')
parser.add_argument('--gamma', type=float, default=0.01,
                    help='number .')
parser.add_argument('--softmax_vectors', type=int, default=6,
                    help='number .')

parser.add_argument('--ROI', type=int, default=90, help='....')

#attention
parser.add_argument('--dims', type=int, default=256,
                    help='....')
parser.add_argument('--heads', type=int, default=4,
                    help='....')