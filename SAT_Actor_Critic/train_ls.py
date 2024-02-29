import argparse
import logging
import os
import pdb
import random
import datetime
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from local_search import WalkSATLN
from warm_up import WarmUP
from utils import *



def train_policy(ls, optimizer, noise_optimizer, critic_optimizer, train_ds, val_ds, args, best_median_flips, model_files, start_time):
    best_epoch = 0
    torch.save(ls.policy.state_dict(), model_files[0])
    if ls.train_noise:
        torch.save(ls.noise_policy.state_dict(), model_files[1])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, steps_per_epoch=1, epochs=args.epochs,
        div_factor=5, final_div_factor=10)
    scheduler_critic = torch.optim.lr_scheduler.StepLR(critic_optimizer, step_size=5, gamma=0.1)
    for i in range(1, args.epochs + 1):
        print("epoch ", i)
        flips, loss, accuracy, critic_losses = ls.train_epoch(optimizer, noise_optimizer, critic_optimizer, scheduler_critic, train_ds)
        to_log(flips, loss, accuracy, start_time, comment="Train Ep " + str(i))
        scheduler.step()
        if i%5 == 0 and i > 0:
            med_flips, mean_flips, accuracy = ls.evaluate(val_ds)
            to_log_eval(med_flips, mean_flips, accuracy, "EVAL  Ep " + str(i))
            if best_median_flips > np.median(med_flips):
                torch.save(ls.policy.state_dict(), model_files[0])
                if ls.train_noise:
                    torch.save(ls.noise_policy.state_dict(), model_files[1])
                best_median_flips = np.median(med_flips)
                best_epoch = i
            if ls.train_noise:
                [w, b] = [p.detach().numpy() for p in ls.noise_policy.parameters()]
                logging.info("parms [{:.2f} {:.2f}] {:.2f}".format(w[0][0], w[0][1], b[0]))
    formatting = 'Best Flips Med: {:.2f}, Best epoch: {}'
    text = formatting.format(best_median_flips,  best_epoch)
    logging.info(text)

def train_warm_up(policy, noise_policy, critic, optimizer, critic_optimizer, train_ds, max_flips=5000):
    wup = WarmUP(policy, noise_policy, critic, max_flips=max_flips)
    for i in range(args.warm_up):
        loss, critic_loss, flips = wup.train_epoch(optimizer, critic_optimizer, train_ds)
        logging.info('Warm_up train loss {:.2f}, critic loss {:.2f}, med flips {:.2f}'.format(loss, critic_loss, flips))

def create_filenames(args):
    model_files = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    basename = args.dir_path.replace("../", "").replace("/", "_") + "_d_" +  str(args.discount)
    basename += "_e" + str(args.epochs) + "_n_" + str(args.n_train)
    if args.train_noise:
        basename += "tr_noise"
    if args.warm_up == 0:
        basename += "_no_wup"
    basename += "_" + timestamp
    log_file = "logs/" + basename +  ".log"
    model_files.append("models/" + basename +  "_score.pt")
    model_files.append("models/" + basename +  "_p.pt")
    return log_file, model_files

def main(args):
    start_time = time.time()

    if args.seed > -1:
        random.seed(args.seed)

    p = get_p(args.dir_path)
    print(p)

    log_file, model_files = create_filenames(args)
    print(log_file, model_files[0])
    if args.train_noise:
        print(model_files[1])

    logging.basicConfig(filename=log_file, level=logging.INFO)

    data = load_dir(args.dir_path)
    train_ds, val_ds = split_data(data, args.n_train)

    policy = Net2(input_features=5)
    optimizer = optim.AdamW(policy.parameters(), lr=args.lr/3, weight_decay=1e-5)
    critic = Critic()
    critic_optimizer = optim.AdamW(critic.parameters(), lr=args.lr_c, weight_decay=1e-5)
    noise_policy = NoiseNet()
    noise_optimizer = optim.AdamW(noise_policy.parameters(), lr=1e-3, weight_decay=1e-5)

    if args.warm_up > 0:
        train_warm_up(policy, noise_policy, critic, optimizer, critic_optimizer, train_ds)
    ls = WalkSATLN(policy, noise_policy, critic, args.train_noise, args.max_tries, args.max_flips, discount=args.discount, gamma=args.gamma, p=p)
    med_flips, mean_flips, accuracy = ls.evaluate(val_ds)
    to_log_eval(med_flips, mean_flips, accuracy, "EVAL No Train/ WarmUP")
    best_median_flips = np.median(med_flips)

    train_policy(ls, optimizer, noise_optimizer, critic_optimizer, train_ds, val_ds, args, best_median_flips, model_files, start_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir_path', type=str)
    parser.add_argument('-m', '--model_path', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_tries', type=int, default=10)
    parser.add_argument('--max_flips', type=int, default=10000)
    parser.add_argument('--p', type=float, default=0.12)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_c', type=float, default=0.01)
    parser.add_argument('--discount', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--warm_up', type=int, default=10)
    parser.add_argument('--n_train', type=int, default=1900)
    parser.add_argument('--train_noise', type=bool, default=False)
    args = parser.parse_args()
    main(args)