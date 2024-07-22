import os
import math
import time
import random
import warnings
import numpy as np
import scipy.sparse as sp
from itertools import chain
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

import utils
import enumeration
import evaluation
import set2box

EPS = 1e-10

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
warnings.filterwarnings('ignore')
args = utils.parse_args()

# GPU Settings ##########
if torch.cuda.is_available():
    device = torch.device("cuda:" + args.gpu)
else:
    device = torch.device("cpu")
print('Device:\t', device, '\n')

# Random Seed ##########
SEED = 2022
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Read data ##########
runtime = time.time()

num_sets, sets, S, M = {}, {}, {}, {}

for dType in ['train', 'valid', 'test']:
    num_items, num_sets[dType], sets[dType] = utils.read_data(args.dataset, dtype=dType)
    S[dType], M[dType] = utils.incidence_matrix(sets[dType], num_sets[dType])
    print('{}:\t|V| = {}\t|E| = {}\t'.format(dType, num_items, num_sets[dType]))

memory = {
    'train': (args.dim * 32 * 2 * num_sets['train']) / 8000,
    'valid': (args.dim * 32 * 2 * num_sets['valid']) / 8000,
    'test': (args.dim * 32 * 2 * num_sets['test']) / 8000
}
    
print('Reading data done:\t\t{} seconds'.format(time.time() - runtime), '\n')

# Enumerate Triplets ##########
start_time = time.time()

enumeration = enumeration.Enumeration(sets['train'])
instances, overlaps = enumeration.enumerate_instances(args.pos_instance, args.neg_instance)

instances, overlaps = instances, overlaps

print('# of instances:\t\t', len(instances))
print('Enumerating pairs done:\t\t', time.time() - start_time, '\n')

# Prepare Evaluation ##########
start_time = time.time()

evaluation = evaluation.Evaluation(sets, isMultitask=False)

print('Preparing evaluation done:\t', time.time() - start_time, '\n')

# Prepare Training ##########
start_time = time.time()

model = set2box.model(num_items, args.dim, args.beta).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

print('Preparing training done:\t', time.time() - start_time, '\n')

# Train Model ##########
S['train'] = S['train'].to(device)
M['train'] = M['train'].to(device)
overlaps = overlaps.to(device)

total_time = time.time()
epoch_losses = []
for epoch in range(1, args.epochs + 1):
    print('\nEpoch:\t', epoch)
    
    # Train ##########

    train_time = time.time()
    model.train()
    epoch_loss = 0
    
    batches = utils.generate_batches(len(instances), args.batch_size)
        
    for i in trange(len(batches), position=0, leave=False):
        # Forward
        batch_instances = instances[batches[i]]
        loss = model.forward(S['train'], M['train'], batch_instances, overlaps[batches[i]])
        epoch_loss += loss.item()
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        model.radius_embedding.weight.data = model.radius_embedding.weight.data.clamp(min=EPS)

    print('Loss:\t', epoch_loss, '\n')
    epoch_losses.append(epoch_loss)
    train_time = time.time() - train_time
    print('train time:{}\t\t seconds'.format(train_time), '\n')

    # Evaluate the Model ##########
    for dType in ['train', 'valid', 'test']:
        eva_time = time.time()
        pred = evaluation.pairwise_similarity(model, S[dType], M[dType], args.beta, dType, args.dim, True)
        for metric in ['oc', 'ji', 'cs', 'di']:
            mse = mean_squared_error(pred[metric], evaluation.ans[dType][metric])
            print('{}_{} (MSE):\t'.format(dType, metric) + str(mse))
        # print('evaluation time:\t\t{} seconds'.format(time.time()-eva_time), '\n')
print('total train time:\t\t{} seconds'.format((time.time() - total_time)), '\t')
print('average train time:\t\t{} seconds'.format((time.time() - total_time)/args.epochs), '\t')
print('losses:', '\t')
for loss in epoch_losses:
    print(str(loss), '\t')

output_data_name = args.dataset + 'N'
embeds = evaluation.get_embeds(model, S['test'], M['test'])
utils.write_embed(output_data_name, embeds, args.dim, args.learning_rate)
