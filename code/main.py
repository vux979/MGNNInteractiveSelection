from dataloader import Dataset
import argparse
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split

from train import train
import torch
import random
import numpy as np
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='book-crossing', help='which dataset to use')
parser.add_argument('--rating_file', type=str, default='implicit_ratings.csv', help='which dataset to use')
parser.add_argument('--dim', type=int, default=64, help='dimension of entity and relation embeddings')
parser.add_argument('--l2_weight', type=float, default=1e-5, help='weight of the l2 regularization term')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--n_epoch', type=int, default=50, help='the number of epochs')
parser.add_argument('--hidden_layer', type=int, default=64, help='neural hidden layer')
parser.add_argument('--num_user_features', type=int, default=-1, help='the number of user attributes')
parser.add_argument('--random_seed', type=int, default=2019, help='size of common item be counted')
parser.add_argument('--heads', type=int, default=3, help='multi-heads')
parser.add_argument('--layer', type=int, default=3, help='GNN layer')
parser.add_argument('--rho', type=float, default=0.5, help='GNN layer')
parser.add_argument('--temp', type=float, default=0.15, help='temperature of InfoNCELoss')
parser.add_argument('--topk', type=float, default=0.3, help='k prop')
parser.add_argument('--eps', type=float, default=0.2, help='perturbed')
parser.add_argument('--cl', type=int, default=1, help='Whether to use contrastive learning')
parser.add_argument("--fl", type=int, default=1, help='Is fft filtering inverse fft used')
#random.choice(range(2000))
args = parser.parse_args()

sep = ','

if args.dataset == 'ml-1m':
    args.num_user_features = 4
elif args.dataset == 'book-crossing':
    args.num_user_features = 3 
elif args.dataset == 'taobao':
    args.num_user_features = 8 
else:
    print("please specify the number of user features: num_user_features")
    


#dataset = Dataset_larger('../data/', args.dataset, args.rating_file, sep, 10)
dataset = Dataset('../data/', args.dataset, args.rating_file, sep, args)

data_num = dataset.data_N()
feature_num = dataset.feature_N()
train_index, val_index = dataset.stat_info['train_test_split_index']
#print(np.concatenate((dataset[0][0:5], dataset[0][6:10])))

# split inner graphs
train_dataset = dataset[:train_index]
val_dataset = dataset[train_index:val_index]
test_dataset = dataset[val_index:]

n_workers = 0
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=n_workers)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=n_workers)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=n_workers)

show_loss = False
print(f"""
datast: {args.dataset}
vector dim: {args.dim}
batch_size: {args.batch_size}
lr: {args.lr}
heads: {args.heads}
layer: {args.layer}
rho: {args.rho}
temp: {args.temp}
topk: {args.topk}
eps: {args.eps}
cl: {args.cl}
fl: {args.fl}
""")
logname = 'last_h'+str(args.heads)+'l'+str(args.layer)+'dim'+str(args.dim) + 'XSimGCL'
logging.basicConfig(filename=f'./log/{args.dataset}/{logname}.log', level=logging.DEBUG)
logging.info(f"""
datast: {args.dataset}
vector dim: {args.dim}
batch_size: {args.batch_size}
lr: {args.lr}
heads: {args.heads}
layer: {args.layer}
rho: {args.rho}
temp: {args.temp}
topk: {args.topk}
eps: {args.eps}
cl: {args.cl}
fl: {args.fl}
"""
)

datainfo = {}
datainfo['train'] = train_loader
datainfo['val'] = val_loader 
datainfo['test'] = test_loader
datainfo['feature_num'] = feature_num 
datainfo['data_num'] = [len(train_dataset), len(val_dataset), len(test_dataset)]


train(args, datainfo, show_loss)

