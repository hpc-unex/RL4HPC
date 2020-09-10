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




# Policy Network

class PolicyNetwork(nn.Module):
	
	def __init__(self, num_inputs, num_outputs, params_nn):
	
		super(PolicyNetwork, self).__init__()
		
		hidden = params_nn["hidden"]
		
		self.num_outputs = num_outputs
		self.hidden1 = nn.Linear(num_inputs, hidden[0])
		self.hidden2 = nn.Linear(hidden[0],  hidden[1])
		self.output  = nn.Linear(hidden[1],  num_outputs)
		
		self.softmax = nn.Softmax(dim = -1)
		
		
	def forward(self, state):
		x = self.hidden1(state)
		x = nn.Dropout(p=0.3)(x)
		x = F.relu(x)
		x = self.hidden2(x)
		x = nn.Dropout(p=0.3)(x)
		x = F.relu(x)
		x = self.output(x)
		x = self.softmax(x)
		
		return x




# Agent

class Agent(object):
	
	def __init__ (self, env, params):
	
		self.params = params
		print(self.params)
		
		self.gamma    = params["gamma"]    # Almost Undiscounted
		self.alpha    = params["alpha"]    # Learning rate (antes: 0.002)
		self.verbose  = params["verbose"]  # Verbosity
		self.P        = params["P"]
		
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
		
		
	def get_return(self):

		T = len(self.saved_rewards)

		# print("SAVED_REWARDS: ", self.saved_rewards)

		# Not to divide by 0:
		eps = np.finfo(np.float32).eps.item()
			
		# Multiply by gamma
		# returns = np.array([ (self.gamma**t) * self.saved_rewards[t] for t in range(T) ])
		#
		# returns = returns[::-1].cumsum()
		# returns = returns[::-1]
		
		# Cummulative sum
		R = 0
		returns = []
		for r in self.saved_rewards[::-1]:
			R = r + self.gamma * R
			returns.insert(0, R)
		returns = np.array(returns)

		# returns = (returns - returns.mean()) # / (returns.std() + eps)

		return returns
	
	
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
				
		if self.verbose:
			print("-----  Select Action  -----")
			print("ACTION_PROBS: ", action_probs, action_probs.shape)
			print("ACTION.ITEM: ",  action.item(), P_s, P_r)
			print("LOG_PROB: ",     logprob, logprob.shape)
		
		return P_s, P_r
		
		
		
	def learn (self):

		# Get discounted reward tensor
		discounted_reward = torch.FloatTensor(self.get_return())

		# List of tensors to tensor
		logprob_tensor = torch.cat(self.saved_logprobs)
		
		loss = -logprob_tensor * discounted_reward
		
		cost = loss.mean()
		
		if self.verbose:
			print("-----  Learn  -----")
			print("DISCOUNTED_REWARDS:", discounted_reward, discounted_reward.shape)
			print("LOG_PROBS: ", logprob_tensor, logprob_tensor.shape)
			print("LOSS vector: ", loss, loss.shape)
			print("LOSS: ", cost, cost.shape)

		self.optimizer.zero_grad()
		cost.backward()
		self.optimizer.step()

		return cost
	
	
	def render (self):
	
		if self.verbose:
			print("-----  Agent  -----")
			# print("Actions: ",  self.saved_actions)
			# print("States:  ",  self.saved_states)
			print("Rewards: ",  self.saved_rewards)
			print("Logprobs: ", self.saved_logprobs)


				
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





# Read config from the .json file:
def read_config ():

	config = {}
	config_file = 'config.json'
	if len(sys.argv) == 2:
		config_file = sys.argv[1]

	try:
		with open(config_file, 'r') as js:
			config = json.load(js)

	except EnvironmentError:
		print ('Error: file not found: ', config_file)

	return config






##########   MAIN   ##########


# Read config/params from a JSON file
config = read_config()


# Parameters for the different parts of the app.
params_agent  = config["Agent"]
params_env    = config["Environment"]
params_bench  = config["Benchmark"]
params_output = config["Output"]


# Create Environment instance
env = MPICollsEnv(params=params_env)


# Create Agent instance
agent = Agent(env, params_agent)



# TODO: what parameter does it need here?
NO_EPISODES = params_agent["n_episodes"]
interval = (NO_EPISODES // 20)

IMB     = params_bench["exec"]
IMBOPT  = params_bench["opts"]

graph_file  = params_output["graph_file"]
hosts_file  = params_output["hosts_file"]
output_file = params_output["output_file"]

# TBD: las siguientes lineas pueden sobrar:
# OPTIONS = IMB + " " + IMBOPT
# EXEC = OPTIONS.split()
# params["exec_command"] = EXEC




J_history = []
T_history = []
R_history = []

start = time.time()

for episode in range(NO_EPISODES):
	
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

	J_history.append(J)
	T_history.append(len(agent.saved_rewards))
	R_history.append(sum(agent.saved_rewards))

	agent.render()


	if (episode % interval == 0):

		start = episode - interval
		end   = episode
		n = end - start
		print("\nEpisode ", episode," of ", NO_EPISODES)
		print("-------------------------")
		print("Loss:    ", sum(J_history[start:end]) / n)
		print("T:       ", sum(T_history[start:end]) / n)
		# print("Depth:   ", np.mean(D_history[start:end]))
		print("Rewards: ", sum(R_history[start:end]) / n)
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
		
		env.render()


end = time.time()
print("Wallclock time: ", end - start)


plot_loss (J_history, T_history)


# Example: predict a trajectory
agent.predict_trajectory()

