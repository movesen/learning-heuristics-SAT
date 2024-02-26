#!/usr/bin/env python
# coding: utf-8

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from local_search import SATLearner


class WarmUP(SATLearner):
    def __init__(self, policy, noise_policy, critic, max_flips=10000, p=0.5):
        super().__init__(policy, noise_policy, critic, max_flips, p)
        self.break_histo = np.zeros(1000)
    
    def get_reward(self, previous_unsat_clauses, current_unsat_clauses):
        reward = previous_unsat_clauses - current_unsat_clauses
        return reward
        
    def select_variable_reinforce(self, x, f, unsat_clause):
        index, lit = self.walksat_step(f, unsat_clause)
        logit = self.policy(x)
        log_prob = F.log_softmax(logit, dim=0)
        return index, log_prob[index]
    
    def reinforce_step(self, f, unsat_clause):
        x = self.stats_per_clause(f, unsat_clause)
        x = torch.from_numpy(x).float()
        index, log_prob = self.select_variable_reinforce(x, f, unsat_clause)
        literal = unsat_clause[index]
        lit_x = x[index]
        critic_lit = torch.tensor([literal], dtype=torch.float32)
        current_value = self.critic_estimate_value(lit_x, critic_lit)
        return literal, log_prob, x, current_value

    def critic_estimate_value(self, x, literal):
        value = self.critic(x, literal)
        return value
    
    def generate_episode_reinforce(self, f):
        self.sol = [x if random.random() < 0.5 else -x for x in range(f.n_variables + 1)]
        self.true_lit_count = self.compute_true_lit_count(f.clauses)
        self.age = np.zeros(f.n_variables + 1)
        self.age2 = np.zeros(f.n_variables + 1)
        log_probs = []
        state_action_pairs = []
        self.flips = 0
        self.flipped = set()
        self.backflipped = 0
        while self.flips < self.max_flips:
            unsat_clause_indices = [k for k in range(len(f.clauses)) if self.true_lit_count[k] == 0]
            current_unsat_clauses = len(unsat_clause_indices)            
            sat = not unsat_clause_indices
            if sat:
                break
            unsat_clause = f.clauses[random.choice(unsat_clause_indices)]
            self.flips +=1
            literal, log_prob, log_prob_p, x, value = self.select_literal(f, unsat_clause)           
            if log_prob:
                log_probs.append(-log_prob)
            self.update_stats(f, literal)
            new_unsat_clauses = len([k for k in range(len(f.clauses)) if self.true_lit_count[k] == 0])
            reward = self.get_reward(current_unsat_clauses, new_unsat_clauses)
            if value:
                state_action_pairs.append((value-reward)**2)
        loss = 0
        cl = 0
        if len(state_action_pairs) > 0:
            cl = torch.mean(torch.stack(state_action_pairs))
        if len(log_probs) > 0:
            loss = torch.mean(torch.stack(log_probs))
        return sat, self.flips, self.backflipped, loss, cl
    
    def evaluate(self, data):
        all_flips = []
        self.policy.eval()
        for f in data:
            sat, flips, backflipped, loss = self.generate_episode_reinforce(f)
            all_flips.append(flips)
        all_flips = np.array(all_flips).reshape(-1)
        return np.median(all_flips), np.mean(all_flips)
        
    def train_epoch(self, optimizer, critic_optimizer, data):
        losses = []
        flip_list = []
        critic_losses = []
        for f in data:
            self.policy.train()
            self.critic.train()
            sat, flips, backflipped, loss, cl = self.generate_episode_reinforce(f)
            flip_list.append(flips)
            if cl:
                critic_optimizer.zero_grad() 
                cl.backward()
                critic_optimizer.step()
                critic_losses.append(cl.item())
            if loss > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
        return np.mean(losses), np.mean(critic_losses), np.median(flip_list)