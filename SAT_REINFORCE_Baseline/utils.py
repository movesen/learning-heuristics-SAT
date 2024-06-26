import argparse
import logging
import os
import pdb
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path

from cnf import CNF


p_dict = {"domset":0.0693, "kclique":0.1218,  "kcolor":0.0895, "rand3sat":0.06436, "rand4sat":0.057}

def model_to_numpy(model):
    parms = [p for p in model.parameters()]
    w = parms[0][0].detach().numpy()
    b = parms[1][0].detach().numpy()
    return (w, b)

def get_p(dir_path):
    dir_path = Path(dir_path)
    dist_name = (dir_path.parent).stem
    return p_dict.get(dist_name, 0.12)

class Net(nn.Module):
    def __init__(self, input_features=4, hidden=10):
        super(Net, self).__init__()
        self.lin = nn.Linear(input_features, hidden)
        self.dropout = nn.Dropout(0.3)
        self.lin2 = nn.Linear(hidden, 1)
    
    def forward(self, x):
        x = self.lin(x)
        x = F.relu(self.dropout(x))
        x = self.lin2(x)
        return x

class NoiseNet(nn.Module):
    def __init__(self, input_features=2):
        super(NoiseNet, self).__init__()
        self.lin = nn.Linear(input_features, 1)
        self.lin.weight.data.uniform_(.1, .5)
        self.lin.bias.data.fill_(0.01)
    def forward(self, x):
        x = self.lin(x)
        return 0.5*torch.sigmoid(x)

class Net2(nn.Module):
    def __init__(self, input_features=4):
        super(Net2, self).__init__()
        self.lin = nn.Linear(input_features, 1)
    def forward(self, x):
        x = self.lin(x)
        return x

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        
        self.branch1 = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU()
        )
        
        self.branch2 = nn.Sequential(
            nn.Linear(1, 5),
            nn.ReLU(),
            nn.Linear(5, 5),
            nn.ReLU()
        )
        
        self.final_layers = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
    
    def forward(self, x1, x2):
        x1 = self.branch1(x1)
        x2 = self.branch2(x2)
        
        x = torch.cat((x1, x2), dim=-1)
        
        output = self.final_layers(x)
        return output

def load_dir(path):
    data = []
    for filename in os.listdir(path):
        name, ext = os.path.splitext(filename)
        if ext != '.cnf':
            continue
        f = CNF.from_file(os.path.join(path, filename))
        data.append(f)
    return data

def split_data(data, N=None, num_vals=100):
    logging.info("length of data is " + str(len(data)))
    if N is None:
        N = len(data) - num_vals
    if len(data) < 100:
        return data, data
    train_ds = data[:N]
    val_ds = data[-num_vals:]
    logging.info("len of train, val " + str(len(train_ds)) + "," + str(len(val_ds)))
    return train_ds, val_ds

def change_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr

def compute_mean_median_CI(med_values, mean_values):
    N = len(med_values)
    medians = [np.median(np.random.choice(med_values, N)) for i in range(1000)]
    means = [np.mean(np.random.choice(mean_values, N)) for i in range(1000)]
    return np.quantile(means, q=[.025, .975]), np.quantile(medians, q=[.025, .975])

def median_CI_2(x):
    """  Practical Nonparametric Statistics, 3rd Edition by W.J. Conover.
    """
    n = len(x)
    q = 0.5
    z = 1.96
    nq = n*q
    j =  int(np.ceil(nq - z*np.sqrt(nq*(1-q))))
    k =  int(np.ceil(nq + z*np.sqrt(nq*(1-q))))
    m = np.sort(x)
    return m[j], np.median(x), m[k]

def compute_median_per_obs(flips, max_tries):
    return [np.median(flips[i:i+max_tries]) for i in range(len(flips)//max_tries)]

def to_log_eval(med_flips, mean_flips, accuracy, comment, CI=False):
    formatting = '{} Flips Med: {:.2f}, Mean: {:.2f} Acc: {:.2f}'
    text = formatting.format(comment, np.median(med_flips), np.mean(mean_flips), 100 * accuracy)
    logging.info(text)
    if CI:
        
        ci_means, ci_median = compute_mean_median_CI(med_flips, mean_flips)
        formatting = 'CI means FLIPS ({:.2f}, {:.2f}), CI median ({:.2f}, {:.2f})'
        text = formatting.format(ci_means[0], ci_means[1], ci_median[0], ci_median[0])
        logging.info(text)
        l, m, h = median_CI_2(med_flips)
        formatting = 'CI median FLIPS low {:.2f}, med {:.2f} high {:.2f}'
        text = formatting.format(l, m, h)
        logging.info(text)


def to_log(flips, loss, accuracy, start_time, comment):
    """when max_tries is not None we compute median flips per observation"""
    if loss is None:
        loss = -1
    end_time = time.time()
    execution_time_seconds = end_time - start_time
    hours, remainder = divmod(execution_time_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatting = '{} Flips Med: {:.2f}, Mean: {:.2f}  Acc: {:.2f} Loss: {:.2f} Exec Time: {:02d}:{:02d}:{:02d}'
    text = formatting.format(comment, np.median(flips), np.mean(flips), 100 * accuracy, loss, int(hours), int(minutes), int(seconds))
    logging.info(text)
