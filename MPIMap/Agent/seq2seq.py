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

# Encoder of the seq2seq policy Network
class Encoder(nn.Module):
	def __init__(self, params, num_inputs, num_hidden):
		super(Encoder, self).__init__()

		self.policy_params = params["Policy"]
		self.typecell = self.policy_params["typecell"]

		self.num_inputs= num_inputs
		self.num_hidden = num_hidden
		if self.typecell == "LSTM":
			self.cell = nn.LSTM(input_size = num_inputs, hidden_size = num_hidden, batch_first = True)
		elif self.typecell == "GRU":
			self.cell = nn.GRU(input_size = num_inputs, hidden_size = num_hidden, batch_first = True)
		else:
			print("ERROR: type of RNN cell not supported")
			return


	def forward(self, state, hidden):
		'''print("State :",state, state.size())
		print("---------------------------------------------------------------")
		print("")
		state = state.long()
		state = state.squeeze(0)
		print("State :",state, state.size())
		print("---------------------------------------------------------------")
		print("")
		embedded = self.embedding(state)
		print("Embedded: ", embedded, embedded.size())
		print("---------------------------------------------------------------")
		print("")'''

		# print("ENCODER (in):  ", state.size(), hidden[0].size())
		output, hidden = self.cell(state, hidden)
		# print("ENCODER (out): ", output.size(), hidden[0].size())

		return  output, hidden


	# def init_state (self, agent, batch_size=1, seq_len=1):
	#
	# 	if self.typecell == "GRU":
	# 		return ( torch.zeros(batch_size, seq_len, self.num_hidden).to(agent.device) )
	# 	elif self.typecell == "LSTM":
	# 		return ( torch.zeros(batch_size, seq_len, self.num_hidden).to(agent.device),
	# 		torch.zeros(batch_size, seq_len, self.num_hidden).to(agent.device) )
	#
	# 	return None


# Decoder of the seq2seq policy Network
class Decoder(nn.Module):

	def __init__(self,params, num_inputs, num_hidden, num_outputs):

		super(Decoder, self).__init__()

		self.policy_params = params["Policy"]
		self.typecell = self.policy_params["typecell"]

		self.params = params["Graph"]
		self.P = self.params["P"]
		self.M = self.params["M"]

		self.num_inputs = num_inputs
		self.num_hidden = num_hidden
		self.num_outputs = num_outputs
		self.out_embed = self.num_hidden//2

		self.embedding = nn.Embedding(self.num_outputs,self.out_embed)
		self.dropout = nn.Dropout(0.1)
		# self.attn = nn.Linear(self.num_hidden * 2 , max_length )
		self.attn = nn.Linear(self.num_hidden + self.num_inputs, self.P)
		# self.attn_combine = nn.Linear(self.num_hidden * 2, self.num_hidden)
		self.attn_combine = nn.Linear(self.num_hidden + self.num_inputs, self.num_hidden)

		if self.typecell == "LSTM":
			self.cell = nn.LSTM(input_size = self.num_hidden, hidden_size = self.num_hidden, batch_first = True)
		elif self.typecell == "GRU":
			self.cell = nn.GRU(input_size = self.num_hidden, hidden_size = self.num_hidden, batch_first = True)
		else:
			print("ERROR: type of RNN cell not supported")
			return

		self.fc = nn.Linear(self.num_hidden,self.num_outputs)

	def forward(self, state, hidden, enc_outputs):
		#Para hacer el embedding es necesario, pasar el tensor de float a long

		# print("[Decoder] state:  ", state.size())
		# print("[Decoder] hidden: ", hidden[0].size())
		# print("[Decoder] output: ", enc_outputs.size())
		# state = state.long()
		# state = state.squeeze(0)
		# print("    [state]: ", state.size())

		# embedded = self.embedding(state)
		# print("    [embedding]: ", embedded.size())

		# embedded = embedded.view(1,1,-1)
		# print("    [embedding]: ", embedded.size())

		# embedded= self.dropout(embedded)
		# print("    [embedding]: ", embedded.size())

		'''El hidden del LSTM lo creamos como una lista de tensores en el constructor
		Lo apilamos en un solo tensor y obtenemos la primera correspondiente al hidden con h[0]
		'''

		# print("[Decoder] hidden: ", hidden[0])

		if self.typecell == "LSTM":
		 	h = torch.stack(hidden, dim=1).squeeze(0)
		elif self.typecell == "GRU":
			h = hidden
		else:
			print("ERROR: type of RNN cell not supported")

		# print("    [cat]:       ", embedded[0].size(), h[0].size(), torch.cat((embedded[0],h[0]),1).size())
		# print("    [cat]:       ", state[0].size(), h[0].size(), torch.cat((state[0],h[0]),1).size())

		# attn_weights = F.softmax(self.attn(torch.cat((embedded[0],h[0]),1)),dim=1)
		attn_weights = F.softmax(self.attn(torch.cat((state[0],h[0]),1)),dim=1)
		# print("    [attn_weigths]: ", attn_weights.size())
		# print("    [attn_weigths]: ", attn_weights)
		# print("    [enc_outputs]:  ", enc_outputs.size())

		# print("    [attn_weigths]: ", attn_weights.size(), enc_outputs.size())
		attn_applied = torch.bmm(attn_weights.unsqueeze(0), enc_outputs)
		# print("    [attn_applied]: ", attn_applied.size())

		# output = torch.cat((embedded[0], attn_applied[0]),1)
		output = torch.cat((state[0], attn_applied[0]),1)
		# print("    [cat]:       ", output.size())

		output = self.attn_combine(output).unsqueeze(0)
		# print("    [output]:       ", output.size())
		output = F.relu(output)
		# print("    [output]:       ", output.size())

		# print("    [cell] in:        ", output.size(), hidden[0].size())
		output, hidden = self.cell(output, hidden)
		# print("    [cell] out:       ", output.size(), hidden[0].size())
		#No he hecho la capa log_softmax, porque tiende a poner todos los resultados a 0
		logits = self.fc(output)

		# print("    [logits]:     ", logits.size())
		# print("    [hidden]:     ", hidden[0].size())
		return  logits, hidden


# Seq2seq new policiy network
class PolicyNetwork(nn.Module):

	def __init__(self, params):

		super().__init__()

		self.params = params["Graph"]
		self.P = self.params["P"]
		self.M = self.params["M"]

		self.hyperparams = params["Hyperparameters"]
		self.K = self.hyperparams["K"]

		self.policy_params = params["Policy"]
		self.typecell = self.policy_params["typecell"]

		# TODO: temporal parameter reading (hard-coded)
		self.num_inputs  = self.policy_params["n_inputs"]
		self.num_outputs = self.policy_params["n_outputs"]
		self.num_hidden  = self.policy_params["n_hidden"]

		# Optimization on GPUs
		if torch.cuda.is_available():
			self.device = torch.device('cuda')
		else:
			self.device = torch.device('cpu')

		# Encoder and decoder creation.
		self.encoder = Encoder(params,
								self.num_inputs,
								self.num_hidden
							  ).to(self.device)

		self.decoder = Decoder(params,
								self.M,            #input (???)
								self.num_hidden,
								self.num_outputs   #Output (??? max_length?)
							  ).to(self.device)


	def forward(self, state):
		#Vector acumulativo de las salidas
		ouputs = torch.zeros(1,1,self.M)
		#Vector inicial para el decoder
		input = torch.zeros(1,1,self.M)
		#Connect encoder hidden with the decoder hidden

		# print("[Policy] state:  ", state, state.size())
		# print("[Policy] hidden: ", self.hidden[0], self.hidden[0].size())
		encoder_output, encoder_hidden = self.encoder(state, self.hidden)

		# print("[Policy] output (out): ", encoder_output, encoder_output.size())
		# print("[Policy] hidden (out): ", encoder_hidden[0], encoder_hidden[0].size())

		flag=False

		for x in range(0, self.P):

			# print("DECODER (in):  ", x, input.size(), encoder_hidden[0].size(), encoder_output.size())
			output, encoder_hidden = self.decoder(input, encoder_hidden, encoder_output)

			# print("DECODER (out): ", x, output.size(), encoder_hidden[0].size())

			#Sólo asignamos una vez para que no se sobreescriban los datos siguientes
			if(flag==False):
				outputs = output
				flag=True

			else:
				#Concatenamos los resultados para los distintos procesos
				outputs = torch.cat((outputs,output), 1)

			input = output
			#Utilizamos la salida como siguiente entrada

		# print("[Policy] outputs: ", outputs, outputs.size())
		return outputs


	def init_state (self):

		batch_sz = self.K
		n_layers = 1

		if self.typecell == "GRU":
			self.hidden = ( torch.zeros(n_layers, batch_sz, self.num_hidden).to(self.device) )
		elif self.typecell == "LSTM":
			self.hidden = ( torch.zeros(n_layers, batch_sz, self.num_hidden).to(self.device),
							torch.zeros(n_layers, batch_sz, self.num_hidden).to(self.device) )
