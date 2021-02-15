#!/usr/bin/env python

## REINFORCE agent

import numpy  as np
import decimal

import os
import sys
sys.path.append('../Env')
sys.path.append('../utils')

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from torch.distributions import Categorical, Multinomial

import time
import pdb
import json

from environ    import MPIMapEnv
from graph      import adjacency
# from rnn        import PolicyNetwork
from seq2seq    import PolicyNetwork



# Agent

class Agent(object):

	def __init__ (self, env, params):

		# Communication graph
		self.graph    = params["Graph"]
		self.P        = self.graph["P"]
		self.root     = self.graph["root"]
		self.M        = self.graph["M"]
		self.comms    = adjacency(self.P, self.graph["comms"])
		self.capacity = torch.tensor(self.graph["capacity"], dtype=torch.long).detach()

		# print("[Agent] Comms:")
		# print(self.comms)
		# print("[Agent] Capacity:")
		# print(self.capacity)

		# Configuration parameters
		self.config        = params["Config"]
		self.rw_type       = self.config["reward_type"]
		self.baseline_type = self.config["Baseline"]
		self.verbose       = self.config["verbose"]     # Verbosity
		self.verbosity_int = self.config["verbosity_interval"]

		# Hyperparameters
		self.hyperparams = params["Hyperparameters"]
		self.gamma       = self.hyperparams["gamma"]       # Discounted factor
		self.alpha       = self.hyperparams["alpha"]       # Learning rate
		self.n_episodes  = self.hyperparams["n_episodes"]
		self.K           = self.hyperparams["K"]           # Num. samples


		# Store loss and episode length evolution
		self.episode     = 0
		self.t           = 0

		self.env = env

		# Optimization on GPUs
		if torch.cuda.is_available():
			self.device = torch.device('cuda')
		else:
			self.device = torch.device('cpu')

		# Parametrized Policy network
		self.policy = PolicyNetwork(params).to(self.device)
		print("Printing the policy", self.policy)
		self.optimizer = torch.optim.Adam(self.policy.parameters(),
		    						  	  lr=self.alpha)

		# Information in each episode
		self.info = {}


	def reset(self):

		# Policy network
		self.policy.train()
		self.policy.init_state()

		# Reset episode
		self.saved_states   = []
		self.saved_actions  = []
		self.saved_rewards  = []
		self.saved_logprobs = []
		self.saved_info     = []

		self.episode += 1
		self.t        = 0


	def save_step(self, s, a, r, info):

		self.saved_states.append(s)
		self.saved_actions.append(a)

		self.saved_rewards = r

		self.info["Episode"]    = self.episode
		self.info["Actions"]    = list(a.detach().numpy())
		self.info["Reward_seq"] = list(r)
		self.info["Reward"]     = info["reward"]
		self.info["Baseline"]   = info["baseline"]
		self.info["Valid"]      = info["valid"]

		self.saved_info.append(info)


	def select_action (self, s):

		# Transform into a Tensor
		state_tensor = torch.FloatTensor(s).unsqueeze(0).to(self.device)

		# Policy: next action
		action_probs = self.policy(state_tensor)
		action_probs = action_probs.view(self.P, -1)


		# Using "masking" of action probabilities
		# Set action_probs to -torch.inf for nodes full
		capacity = self.capacity.clone().detach()
		actions = torch.zeros(self.P, dtype=torch.long)
		logprobs = torch.FloatTensor(self.P)


		# print("[select_action] action_probs: ", action_probs, action_probs.size())
		idx = torch.arange(self.M, dtype=torch.long).to(self.device)
		for i in range(0, self.P):
			p = torch.gather(action_probs[i], 0, idx).to(self.device)

			# print("[select_action] idx: ", idx)
			# print("[select_action] p:   ", p)

			for k in range(0, self.M):
				if capacity[k] == 0:
					p[k] = -np.inf

			# print("[select_action] p:   ", p)

			a_dist = Categorical(logits=p)
			a = a_dist.sample()
			actions[i] = a
			logprobs[i] = a_dist.log_prob(a)

			capacity[a] -= 1

			self.t += 1

			# print("[select_action] actions:   ", actions)


		# Using a Multinomial distribution:
		"""
		a_dist  = Multinomial(logits=action_probs)
		actions = a_dist.sample()
		logprobs = a_dist.log_prob(actions)
		actions = actions.argmax(dim=1)
		"""

		# Using a Categorical distribution:
		"""
		a_dist  = Categorical(logits=action_probs)
		actions = a_dist.sample()
		logprobs = a_dist.log_prob(actions)
		"""


		self.saved_logprobs.append(logprobs)

		return actions


	def get_return(self, rewards):

		T = len(rewards)
		returns = torch.zeros(T)

		# Cummulative sum of rewards (G_t)
		future_ret = 0
		for t in reversed(range(T)):
			future_ret = rewards[t] + self.gamma * future_ret
			returns[t] = future_ret

		# returns = (returns - returns.mean()) / returns.std()

		return returns


	def learn (self):

		# Compute discounted rewards (to Tensor):
		# print("[REINFORCE] rewards: ", self.saved_rewards)
		discounted_reward = self.get_return(self.saved_rewards).to(self.device)
		# print("[REINFORCE] discounted_reward: ", discounted_reward)

		# Compute log_probs:
		logprob_tensor = torch.stack(self.saved_logprobs).to(self.device)
		# print("[REINFORCE] logprob: ", logprob_tensor)

		# Compute loss:
		loss = -logprob_tensor * discounted_reward
		# print("[REINFORCE] v. loss: ", loss)
		loss = torch.sum(loss)
		# print("[REINFORCE] loss: ", loss)

		# Update parameters:
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		# Complete information
		self.info["Discounted rw"]  = list(discounted_reward.detach().numpy())
		self.info["Logprobs"] = list(logprob_tensor.detach().squeeze().numpy())
		self.info["J"]  = loss.item()
		self.info["T"]  = self.t

		return self.info


	def predict_trajectory (self):

		s = self.env.reset() # initial state

		terminal = False

		self.policy.eval()

		while not terminal:

			# Transform into a Tensor
			state_tensor = torch.FloatTensor(s).unsqueeze(0).to(self.device)

			# Policy: next action
			action_probs = self.policy(state_tensor)
			action_probs = action_probs.view(self.P, -1)

			a = action_probs.argmax(dim=1)

			s_, r, terminal, info = self.env.step(a, True)
			s = s_

		return a, action_probs, r
