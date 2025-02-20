import os
import math
import time
import random
import warnings
import numpy as np
import scipy.sparse as sp
from itertools import chain

from torch import autograd
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from early_stopping import EarlyStopping

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils
import enumeration
import evaluation
import set2boxm


EPS = 1e-10

warnings.filterwarnings('ignore')
args = utils.parse_args()

########## GPU Settings ##########
if torch.cuda.is_available():
    device = torch.device("cuda:" + args.gpu)
else:
    device = torch.device("cpu")
print('Device:\t', device, '\n')

########## Random Seed ##########
SEED = 2024
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

########## Read data ##########
runtime = time.time()

num_sets, sets, S, M = {}, {}, {}, {}

for dtype in ['train', 'valid', 'test']:
    num_items, num_sets[dtype], sets[dtype] = utils.read_data(args.dataset, dtype=dtype)
    S[dtype], M[dtype] = utils.incidence_matrix(sets[dtype], num_sets[dtype])
    print('{}:\t\t|V| = {}\t|E| = {}'.format(dtype, num_items, num_sets[dtype]))

memory = {
    'train': ((args.dim * 32 * 2 * args.K) + (num_sets['train'] * args.D * math.log2(args.K))) / 8000,
    'valid': ((args.dim * 32 * 2 * args.K) + (num_sets['valid'] * args.D * math.log2(args.K))) / 8000,
    'test': ((args.dim * 32 * 2 * args.K) + (num_sets['test'] * args.D * math.log2(args.K))) / 8000
}
    
print('Reading data done:\t\t{} seconds'.format(time.time() - runtime), '\n')

########## Enumerate Triplets ##########
start_time = time.time()

enumeration = enumeration.Enumeration(sets['train'])
instances, similarities = enumeration.enumerate_instances_multi(args.pos_instance, args.neg_instance)

instances, similarities = instances, similarities

print('# of instances:\t\t', len(instances))
print('Enumerating pairs done:\t\t', time.time() - start_time, '\n')

########## Prepare Evaluation ##########
start_time = time.time()

evaluation = evaluation.Evaluation(sets, True)

print('Preparing evaluation done:\t', time.time() - start_time, '\n')

########## Prepare Training ##########
start_time = time.time()

model = set2boxm.model(num_items, args.dim, args.beta).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

print('Preparing training done:\t', time.time() - start_time, '\n')

########## Train Model ##########
S['train'] = S['train'].to(device)
M['train'] = M['train'].to(device)
similarities = similarities.to(device)

total_time = time.time()
epoch_agglosses = []
stop_epoch = args.epochs
early_stopping = EarlyStopping(depth=10, ignore=5)
for epoch in range(1, args.epochs + 1):
    print('\nEpoch:\t', epoch)
    print('\n')

    ########## Train ##########
    train_time = time.time()
    model.train()
    if args.lmbda == 0:
        epoch_loss = 0
    else:
        epoch_losses = {1: 0, 2: 0, 3: 0, 4: 0}
    
    batches = utils.generate_batches(len(instances), args.batch_size)

    ## with autograd.detect_anomaly():
    for i in trange(len(batches), position=0, leave=False):
        # Forward
        # L2 = 0
        # for w in model.parameters():
        #     L2 = torch.norm(w, p=2)*1e-8
        batch_instances = instances[batches[i]]
        loss_1, loss_2, loss_3, loss_4 = \
            model.forward(S['train'], M['train'], batch_instances, similarities[batches[i]])
        epoch_losses[1] += loss_1.item()
        epoch_losses[2] += loss_2.item()
        epoch_losses[3] += loss_3.item()
        epoch_losses[4] += loss_4.item()

        if(args.weight == 'EW'):
            loss_agg = 0.25 * (loss_1 + loss_2 + loss_3 + loss_4)
        if(args.weight == 'AD'):
            loss_total = loss_1 + loss_2 + loss_3 + loss_4 + EPS
            loss_w = [loss_1/loss_total, loss_2/loss_total, loss_3/loss_total, loss_4/loss_total]
            loss_agg = loss_w[0] * loss_1 + loss_w[1] * loss_2 + loss_w[2] * loss_3 + loss_w[3] * loss_4
        #if(args.weight == 'mgn'):

        # Optimize
        optimizer.zero_grad()
        loss_agg.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        model.radius_embedding.weight.data = model.radius_embedding.weight.data.clamp(min=EPS)

    aggloss = 0
    if args.lmbda == 0:
        print('Loss:\t', epoch_loss)
    else:
        for i in range(1, 4+1):
            print('Loss {}:\t'.format(i), epoch_losses[i])
            aggloss += epoch_losses[i]
    train_time = time.time() - train_time
    print('train time:{}\t\t seconds'.format(train_time), '\n')
    epoch_agglosses.append(aggloss/4)

    ########## Evaluate the Model ##########
    mse_valid = []
    for dtype in ['train', 'valid', 'test']:
        eva_time = time.time()
        pred = evaluation.pairwise_similarity_multitask(model, S[dtype], M[dtype], args.beta, dtype, args.dim)
        for metric in ['oc', 'ji', 'cs', 'di']:
            mse = mean_squared_error(pred[metric], evaluation.ans[dtype][metric])
            print('{}_{} (MSE):\t'.format(dtype, metric) + str(mse))
            if dtype == 'valid':
                mse_valid.append(mse)
        # print('evaluation time:\t\t{} seconds'.format(time.time() - eva_time), '\n')

    ############# Early Stop ################
    avg_mse_valid = np.average(mse_valid)
    if early_stopping.check(avg_mse_valid):
        stop_epoch = epoch
        print("Early stopping")
        break


print('total train time:\t\t{} seconds'.format((time.time() - total_time)), '\t')
print('average train time:\t\t{} seconds'.format((time.time() - total_time)/epoch), '\t')
print('Stop Epoch: {}'.format(stop_epoch), '\t')
print('losses:', '\t')
for loss in epoch_agglosses:
    print(str(loss), '\t')

output_data_name = args.dataset + 'MT'
embeds = evaluation.get_embeds_multitask(model, S['test'], M['test'])
utils.write_embed(output_data_name, embeds, args.dim, args.learning_rate)
