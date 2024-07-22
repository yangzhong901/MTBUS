import os
import random
import pyfastrand
import numpy as np
import pickle as pkl
from itertools import combinations
from collections import defaultdict
from tqdm import tqdm, trange
import torch

EPS = 1e-10

class Enumeration():
    def __init__(self, sets):
        super().__init__()
        self.E2V = sets
        self.V2E = defaultdict(list)
        self.E2V_set = defaultdict(set)
        self.prepare_data()
        
    def prepare_data(self):
        for i, e in enumerate(self.E2V):
            for v in e:
                self.V2E[v].append(i)
            self.E2V_set[i] = set(e)
                
    def get_instance_info(self, e_i, e_j, e_k):
        I_i = len(self.E2V[e_i])
        I_j = len(self.E2V[e_j])
        I_k = len(self.E2V[e_k])
        I_ij = len(self.E2V_set[e_i].intersection(self.E2V_set[e_j]))
        I_jk = len(self.E2V_set[e_j].intersection(self.E2V_set[e_k]))
        I_ki = len(self.E2V_set[e_k].intersection(self.E2V_set[e_i]))
        I_ijk = len(self.E2V_set[e_i].intersection(self.E2V_set[e_j]).intersection(self.E2V_set[e_k]))

        instance = [e_i, e_j, e_k]
        overlap = [I_i, I_j, I_k, I_ij, I_jk, I_ki, I_ijk]
        overlap = [x / sum(overlap) for x in overlap]
        return instance, overlap
        
    def sample_neighbor(self, e_i):
        v = self.E2V[e_i][pyfastrand.pcg32bounded(len(self.E2V[e_i]))]
        e_j = self.V2E[v][pyfastrand.pcg32bounded(len(self.V2E[v]))]        
        return e_j
    
    def enumerate_instances(self, pos_size, neg_size, rand_seed=2024):
        np.random.seed(rand_seed)
        
        instances, overlaps = [], []

        for e_i in trange(len(self.E2V), position=0, leave=False):
            size_i = len(self.E2V[e_i])

            for _ in range(pos_size):
                e_j = self.sample_neighbor(e_i)
                if pyfastrand.pcg32bounded(2) == 0:
                    e_k = self.sample_neighbor(e_i)
                else:
                    e_k = self.sample_neighbor(e_j)
                instance, overlap = self.get_instance_info(e_i, e_j, e_k)
                instances.append(instance)
                overlaps.append(overlap)

            for _ in range(neg_size):
                e_j = pyfastrand.pcg32bounded(len(self.E2V))
                e_k = pyfastrand.pcg32bounded(len(self.E2V))
                instance, overlap = self.get_instance_info(e_i, e_j, e_k)
                instances.append(instance)
                overlaps.append(overlap)

        instances = torch.tensor(instances)
        overlaps = torch.tensor(overlaps).float()
            
        return instances, overlaps

    def get_instance_info_multi(self, e_i, e_j):
        l_i = len(self.E2V[e_i])
        l_j = len(self.E2V[e_j])
        l_ij = len(self.E2V_set[e_i].intersection(self.E2V_set[e_j]))
        # overlap = l_ij
        overlap = l_ij/max(l_i, l_j)
        jaccard = l_ij/(l_i + l_j - l_ij)
        cosine = l_ij/((l_i*l_j)**0.5)
        dice = 2*l_ij/(l_i + l_j)
        instance = [e_i, e_j]

        sims = [overlap, jaccard, cosine, dice]
        return instance, sims
        #  return instance, overlap, jaccard, cosine, dice

    def enumerate_instances_multi(self, pos_size, neg_size, rand_seed=2024):
        np.random.seed(rand_seed)

        instances, similarities = [], []
        oc, ji, cs, di = [], [], [], []

        for e_i in trange(len(self.E2V), position=0, leave=False):
            size_i = len(self.E2V[e_i])

            for _ in range(pos_size):
                e_j = self.sample_neighbor(e_i)

                # instance, o, j, c, d = self.get_instance_info_multi(e_i, e_j)
                instance, similarity = self.get_instance_info_multi(e_i, e_j)
                instances.append(instance)
                similarities.append(similarity)
                # oc.append(o)
                # ji.append(j)
                # cs.append(c)
                # di.append(d)

            for _ in range(neg_size):
                e_j = pyfastrand.pcg32bounded(len(self.E2V))
                # instance, o, j, c, d = self.get_instance_info_multi(e_i, e_j)
                instance, similarity = self.get_instance_info_multi(e_i, e_j)
                instances.append(instance)
                similarities.append(similarity)
                # oc.append(o)
                # ji.append(j)
                # cs.append(c)
                # di.append(d)

        instances = torch.tensor(instances)
        similarities = torch.tensor(similarities).float()
        # oc = torch.tensor(oc)
        # ji = torch.tensor(ji)
        # cs = torch.tensor(cs)
        # di = torch.tensor(di)


        return instances, similarities
