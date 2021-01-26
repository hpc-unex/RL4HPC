#!/usr/bin/env python

## Parametrized Policy network.

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



# Policy Network

class PolicyNetwork(nn.Module):

	def __init__(self, params):

		super(PolicyNetwork, self).__init__()

		self.params = params["Graph"]
		P = self.params["P"]
		M = self.params["M"]

		self.hyperparams = params["Hyperparameters"]
		self.K = self.hyperparams["K"]

		self.policy_params = params["Policy"]
		self.typecell = self.policy_params["typecell"]

		self.num_inputs  = P * P * 2
		self.num_outputs = M
		self.num_hidden  = P * P * 2


		if self.typecell == "LSTM":
			self.cell = nn.LSTM(input_size=self.num_inputs, hidden_size=self.num_hidden, batch_first=True)
		elif self.typecell == "GRU":
			self.cell = nn.GRU(input_size=self.num_inputs, hidden_size=self.num_hidden, batch_first=True)
		else:
			print("ERROR: type of RNN cell not supported")
			return

		self.fc = nn.Linear(self.num_hidden, self.num_outputs)

		# Optimization on GPUs
		if torch.cuda.is_available():
			self.device = torch.device('cuda')
		else:
			self.device = torch.device('cpu')


	def forward(self, state):

		"""
		print("State tensor: ", state.shape)

		logits = torch.zeros((8, self.num_outputs))

		for i in range(0, 8):
			print("   state process: ", state[:,i,:].view(1,1,self.num_inputs).shape)

			output, hidden = self.cell(state[:,i,:].view(1,1,self.num_inputs), hidden)

			print("       output: ", output.shape)

			logits[i] = self.fc(output)

		print("       logits: ", logits.shape)
		"""

		output, hidden = self.cell(state, self.hidden)

		logits = self.fc(output)

		return logits


	def init_state (self):

		batch_sz = self.K
		n_layers = 1

		if self.typecell == "GRU":
			self.hidden = ( torch.zeros(n_layers, batch_sz, self.num_hidden).to(self.device) )
		elif self.typecell == "LSTM":
			self.hidden = ( torch.zeros(n_layers, batch_sz, self.num_hidden).to(self.device),
							torch.zeros(n_layers, batch_sz, self.num_hidden).to(self.device) )
