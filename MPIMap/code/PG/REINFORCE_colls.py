#!/usr/bin/env python

## MonteCarlo Policy Gradients (REINFORCE)
#
# Implementing MonteCarlo policy gradient algorithm
# (REINFORCE, [Williams, 1992]) to learn a mapping for a
# given communication graph representing an application.

import numpy  as np
import decimal

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

from mpicolls_env  import MPICollsEnv
from graph         import plot_graph
from plots         import plot_loss, plot_file
from config        import read_config



# Policy Network

class PolicyNetwork(nn.Module):

	def __init__(self, params, num_inputs, num_hidden, num_outputs):

		super(PolicyNetwork, self).__init__()

		self.typecell = params["typecell"]

		self.num_inputs  = num_inputs
		self.num_outputs = num_outputs
		self.num_hidden  = num_hidden

		if self.typecell == "LSTM":
			self.cell = nn.LSTM(input_size=num_inputs, hidden_size=num_hidden, batch_first=True)
		elif self.typecell == "GRU":
			self.cell = nn.GRU(input_size=num_inputs, hidden_size=num_hidden, batch_first=True)
		else:
			print("ERROR: type of RNN cell not supported")
			return

		self.fc = nn.Linear(self.num_hidden, self.num_outputs)
		torch.nn.init.xavier_normal_(self.fc.weight)


	def forward(self, state, hidden):

		output, hidden = self.cell(state, hidden)

		logits = self.fc(output)
		# logits = F.softmax(logits, dim=2)

		return logits


	def init_state (self, agent, batch_size=1, seq_len=1):

		if self.typecell == "GRU":
			return ( torch.zeros(batch_size, seq_len, self.num_hidden).to(agent.device) )
		elif self.typecell == "LSTM":
			return ( torch.zeros(batch_size, seq_len, self.num_hidden).to(agent.device),
					 torch.zeros(batch_size, seq_len, self.num_hidden).to(agent.device) )

		return None



# Agent

class Agent(object):

	def __init__ (self, env, params):

		self.params = params

		self.gamma         = params["gamma"]       # Almost Undiscounted
		self.alpha         = params["alpha"]       # Learning rate
		self.P             = params["P"]
		self.M             = params["M"]
		self.n_episodes    = params["n_episodes"]
		self.K             = params["K"]           # Num trajectory samples
		self.baseline_type = params["Baseline"]    # Baseline to use
		self.verbose       = params["verbose"]     # Verbosity
		self.verbosity_int = params["verbosity_interval"]

		# Store loss and episode length evolution
		self.J_history   = []
		self.T_history   = []
		self.n_errors    = []
		self.n_intra     = []
		self.n_inter     = []
		self.episode     = 0
		self.t           = 0

		# Output (training results) file
		params_iodata = params_agent["I/O Data"]
		output_file = params_iodata["output_file"]
		self.o_file = open(output_file, "a")

		# Policy network
		if torch.cuda.is_available():
			self.device = torch.device('cuda')
		else:
			self.device = torch.device('cpu')

		policy_params    = params["NN"]
		self.policy = PolicyNetwork(policy_params,
		                            self.P * self.P * 2,        # Input
									self.P * self.P * 2,        # hidden
								    self.M         # Output
									).to(self.device)

		"""
		for param in self.policy.parameters():
			if len(param.shape) >= 2:
				torch.nn.init.xavier_normal_(param.data)
			else:
				torch.nn.init.normal_(param.data)
		"""


		self.optimizer = torch.optim.Adam(self.policy.parameters(),
		    						  	  lr=self.alpha)


	def reset(self):

		# Policy network
		self.policy.train()
		self.hidden_tensor = self.policy.init_state(agent, batch_size=1, seq_len=1)

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

		self.saved_info.append(info)


	def select_action (self, s):

		# Transform into a Tensor
		state_tensor = torch.FloatTensor([s]).to(self.device)

		# Policy: next action
		action_probs = self.policy(state_tensor, self.hidden_tensor)
		action_probs = action_probs.view(self.P, -1)

		a_dist   = Multinomial(logits=action_probs)
		actions  = a_dist.sample()
		logprobs = a_dist.log_prob(actions)

		actions = actions.argmax(dim=1)

		self.saved_logprobs.append(logprobs)
		# self.saved_logprobs = logprobs

		self.t += 1

		return actions


	def get_return(self):

		T = len(self.saved_rewards)
		returns = np.empty(T, dtype=np.float)

		# Cummulative sum of rewards (G_t)
		future_ret = 0
		for t in reversed(range(T)):
			future_ret = self.saved_rewards[t] + self.gamma * future_ret
			returns[t] = future_ret

		return returns



	def learn (self):

		# Compute discounted rewards (to Tensor):
		discounted_reward = torch.FloatTensor(self.get_return()).to(self.device)
		# print("[REINFORCE] discounted_reward: ", discounted_reward)

		# Compute log_probs:
		logprob_tensor = torch.stack(self.saved_logprobs).to(self.device)
		# logprob_tensor = self.saved_logprobs
		# print("[REINFORCE] logprob: ", logprob_tensor)

		# Compute loss:
		loss = -(logprob_tensor * discounted_reward)
		# print("[REINFORCE] v. loss: ", loss)
		loss = torch.sum(loss)
		# print("[REINFORCE] loss: ", loss)

		# Update parameters:
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		return loss


	def predict_trajectory (self):

		s = env.reset() # initial state
		# agent.reset()

		terminal = False

		self.policy.eval()

		while not terminal:

			# Transform into a Tensor
			state_tensor = torch.FloatTensor([s]).to(self.device)

			# Policy: next action
			action_probs = self.policy(state_tensor, self.hidden_tensor)
			action_probs = action_probs.view(self.P, -1)

			a = action_probs.argmax(dim=1)

			s_, r, terminal, info = env.step(a, True)
			s = s_

		return a, action_probs, r


	def render (self, J):

		# Output file:
		baseline = self.saved_info[-1]["baseline"]
		reward   = self.saved_info[-1]["reward"]
		self.o_file.write(str(self.episode) + " \t " + str(J.item()) + " \t " + str(time.time()) + " \t " + str(self.t) + " \t " + str(reward) + " \t " + str(baseline) + "\n")
		if self.episode == self.n_episodes:
			self.o_file.close()

		# Console output
		info = np.array(self.saved_info)

		self.J_history.append(J.item())
		self.T_history.append(len(self.saved_rewards))

		if self.verbose and not (self.episode % self.verbosity_int):

			start = self.episode - self.verbosity_int
			end   = self.episode
			n     = end - start

			if n != 0:

				costs = np.array(self.J_history[start:end])

				print("\nAGENT: Episode ", self.episode," of ", self.n_episodes)
				print("--------------------------------------")
				print("Loss:       ", sum(self.J_history[start:end]) / n)
				print("Loss Acc:   ", sum(self.J_history) / len(self.J_history))
				print("T:          ", sum(self.T_history[start:end]) / n)
				print("info:       ", self.saved_info[-1])

				if (costs.size > 0):
					print("Loss (mean/std/max/min): ", costs.mean(), costs.std(), costs.max(), costs.min())
					print("Rewards:      ", self.saved_rewards)
					print("Discounted R: ", self.get_return())
					print("log probs:    ", self.saved_logprobs) #torch.stack(self.saved_logprobs).detach().numpy())

					# Show a trajectory:
					a, action_probs, r = self.predict_trajectory()
					print("Probs:   ")
					print(action_probs)
					print("Actions: ", a)
					print("Rewards: ", r)

				print(flush=True)

			return




def print_header (f, pagent, penv):

	f.write("#P: " + str(pagent["P"]) + "\n")
	f.write("#M: " + str(pagent["M"]) + "\n")
	f.write("#alpha: " + str(pagent["alpha"]) + "\n")
	f.write("#gamma: " + str(pagent["gamma"]) + "\n")
	f.write("#n_episodes: " + str(pagent["n_episodes"]) + "\n")
	f.write("#K: " + str(pagent["K"]) + "\n")
	f.write("#Baseline: " + str(pagent["Baseline"]) + "\n")

	f.write("#m: " + str(penv["m"]) + "\n")
	f.write("#Nodes: " + str(penv["nodes"]) + "\n")
	f.write("#Processors/Node: " + str(penv["nodes_procs"]) + "\n")
	f.write("#Reward_type: " + str(penv["reward_type"]) + "\n")
	f.write("#StateRep: " + str(penv["state_rep"]) + "\n")
	f.write("#StartTime: " + str(time.time()) + "\n")
	f.write("# \n")

	f.write("#  e   \t   J   \t  time  \t   T   \t reward \t baseline \n")
	f.write("#----- \t ----- \t ------ \t ----- \t ------ \t -------- \n")




##########   MAIN   ##########

# Set to GPU
if torch.cuda.is_available():
	print("Training model in GPUs ... ")
	torch.cuda.set_device("cuda:0")

# Read config/params from JSON file
config = read_config()

# Parameters for the different parts of the app.
params_agent  = config["Agent"]
params_env    = config["Environment"]


# Output file (header):
params_iodata = params_agent["I/O Data"]
output_file = params_iodata["output_file"]
o_file = open(output_file, "w")
print_header(o_file, params_agent, params_env)
o_file.close()


######################
# Main Learning Loop:
######################

# Create Environment instance
env = MPICollsEnv(params=params_env)
# Create Agent instance
agent = Agent(env, params=params_agent)

start = time.time()

for episode in range(agent.n_episodes):

	terminal = False

	s = env.reset()
	agent.reset()

	while not terminal:

		a = agent.select_action(s)

		s_, r, terminal, info = env.step(a)

		agent.save_step(s, a, r, info)

		s = s_

	# Learn policy
	J = agent.learn()

	# Show partial results
	agent.render(J)
	env.render()


end = time.time()
print("Wallclock time: ", end - start)


# Example: predict a trajectory
a = agent.predict_trajectory()
print(a)
