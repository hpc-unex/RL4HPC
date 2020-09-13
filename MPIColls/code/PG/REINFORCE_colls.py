#!/usr/bin/env python

## MonteCarlo Policy Gradients (REINFORCE)
#
# Implementing MonteCarlo policy gradient algorithm
# (REINFORCE, [Williams, 1992]) to learn an optimal graph for
# performing a bropadcast MPI collective operation.




import numpy  as np

import sys
sys.path.append('../Env')
sys.path.append('../utils')

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from torch.distributions import Categorical

import time
import pdb
import json

from mpicolls_env  import MPICollsEnv
from graph         import plot_graph
from plots         import plot_loss
from config        import read_config




# Policy Network

class PolicyNetwork(nn.Module):
	
	def __init__(self, num_inputs, num_outputs, params_nn):
	
		super(PolicyNetwork, self).__init__()
		
		hidden = params_nn["hidden"]
		
		self.num_outputs = num_outputs
		self.hidden  = nn.Linear(num_inputs, hidden[0])
		self.output  = nn.Linear(hidden[0],  num_outputs)
		
		self.softmax = nn.Softmax(dim = -1)
	
		# self.train()
		
		
	def forward(self, state):
		x = self.hidden(state)
		x = nn.Dropout(p=0.3)(x)
		x = F.relu(x)
		x = self.output(x)
		x = self.softmax(x)
		
		return x




# Agent

class Agent(object):
	
	def __init__ (self, env, params):
	
		self.params = params
		
		self.gamma         = params["gamma"]    # Almost Undiscounted
		self.alpha         = params["alpha"]    # Learning rate (antes: 0.002)
		self.P             = params["P"]
		self.n_episodes    = params["n_episodes"]
		self.verbose       = params["verbose"]  # Verbosity
		self.verbosity_int = params["verbosity_interval"]
		
		# Store loss evolution
		self.J_history = []
		
		self.policyNN = PolicyNetwork(self.P*self.P,  # Input
									  self.P*self.P,  # Output
									  params["NN"])   # Hidden
		
		self.optimizer = torch.optim.Adam(self.policyNN.parameters(),
										  lr=self.alpha)
		
		
	def reset(self):
		# Reset episode
		self.saved_states   = []
		self.saved_actions  = []
		self.saved_rewards  = []
		self.saved_logprobs = []
	
	
	def save_step(self, s, a, r):
		self.saved_states.append(s)
		self.saved_actions.append(a)
		self.saved_rewards.append(r)
		
		
	
	def select_action (self, s):
	
		# Transform into a Tensor
		state_tensor = torch.FloatTensor(s).view(1, -1)
		
		# Policy: next action
		action_probs = self.policyNN(state_tensor)
		
		# Sample
		a_dist  = Categorical(action_probs)
		action  = a_dist.sample()
		logprob = a_dist.log_prob(action)
		
		# Save logprob
		self.saved_logprobs.append(logprob)
		
		# Sender and receiver
		P_s = action.item() // self.P
		P_r = action.item() %  self.P
				
		
		return P_s, P_r
		
		
	def get_return(self):
		
		T = len(self.saved_rewards)
		returns = np.empty(T, dtype=np.float)
		
		# Cummulative sum (G_t)
		future_ret = 0
		for t in reversed(range(T)):
			future_ret = self.saved_rewards[t] + self.gamma * future_ret
			returns[t] = future_ret
		
		
		# TODO: Baseline
		# eps = np.finfo(np.float32).eps.item() # Not to divide by 0
		# returns = (returns - returns.mean()) # / (returns.std() + eps)

		
		return returns

	

	def learn (self):

		# Compute discounted rewards (to Tensor):
		discounted_reward = torch.FloatTensor(self.get_return())

		# Compute log_probs:
		logprob_tensor = torch.cat(self.saved_logprobs)
		
		loss = -logprob_tensor * discounted_reward
		loss = torch.sum(loss)
		
		# Update parameters
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		return loss

	
	def predict_trajectory (self):
		
		s = env.reset() # Start from initial state
		agent.reset()
		terminal = False
		t = 0
		
		print(s)

		while not terminal:
			
			a = self.predict(s)
			
			print(a)
			s_, r, terminal, info = env.step(a)
			s = np.copy(s_)
			
			t = t + 1
			if (t > self.P * 2):  # Error control
				break

		print(s_)
		plot_graph(s_, "Monte Carlo Policy Gradients")


			
	def predict (self, s):

		# Transform into a Tensor
		state_tensor = torch.FloatTensor(s).view(1, -1)

		# Policy: next action
		action_probs = self.policyNN(state_tensor)
		
		# Sample
		action = action_probs.argmax()
		
		# Sender and receiver
		P_s = action.item() // self.P
		P_r = action.item() %  self.P
		
		return (P_s, P_r)


	def render (self, episode, J):
		
		self.J_history.append(J)
		# T_history.append(len(agent.saved_rewards))
		# R_history.append(sum(agent.saved_rewards))
		
		if self.verbose and not (episode % self.verbosity_int):
			
			start = episode - self.verbosity_int
			end   = episode
			n     = end - start
			print("\nAGENT: Episode ", episode," of ", self.n_episodes)
			print("-------------------------")
			print("Loss:    ", sum(agent.J_history[start:end]) / n)
			# print("T:       ", sum(agent.T_history[start:end]) / n)
			# print("Depth:   ", np.mean(D_history[start:end]))
			# print("Rewards: ", sum(R_history[start:end]) / n)
			print(flush=True);
			"""
			o_file = open(output_file, "a")
			o_file.write("Episode " + str(episode) + " of " + str(NO_EPISODES) + "\n")
			o_file.write("------------------------- \n")
			o_file.write("Loss:    " + str(np.mean(J_history[start:end])) + "\n")
			o_file.write("T:       " + str(np.mean(T_history[start:end])) + "\n")
			# o_file.write("Depth:   " + str(np.mean(D_history[start:end])) + "\n")
			o_file.write("Rewards: " + str(np.mean(R_history[start:end])) + "\n")
			o_file.flush();
			o_file.close()
			"""






##########   MAIN   ##########


# Read config/params from a JSON file
config = read_config()

# Parameters for the different parts of the app.
params_agent  = config["Agent"]
params_env    = config["Environment"]
params_iodata = config["I/O Data"]


# Create Environment instance
env = MPICollsEnv(params=params_env)


# Create Agent instance
agent = Agent(env, params_agent)



# TODO: what parameter does it need here?
"""
IMB     = params_bench["exec"]
IMBOPT  = params_bench["opts"]
# TBD: las siguientes lineas pueden sobrar:
# OPTIONS = IMB + " " + IMBOPT
# EXEC = OPTIONS.split()
# params["exec_command"] = EXEC
graph_file  = params_output["graph_file"]
hosts_file  = params_output["hosts_file"]
output_file = params_output["output_file"]
"""



# Main Learning Loop:

start = time.time()

for episode in range(agent.n_episodes):
	
	terminal = False
	reward   = 0
	steps    = 0
				
	s = env.reset()
	agent.reset()
				
	while not terminal:

		a = agent.select_action(s)

		s_, r, terminal, info = env.step(a)

		agent.save_step(s, a, r)
	
		s = np.copy(s_)

	# Learn Policy
	J = agent.learn()

	# Show partial results
	agent.render(episode, J)
	env.render(episode)


end = time.time()
print("Wallclock time: ", end - start)


# Outputs
plot_loss (agent.J_history)


# Example: predict a trajectory
agent.predict_trajectory()

