#!/usr/bin/env python
# coding: utf-8

# ## MonteCarlo Policy Gradients (REINFORCE)
#
# Implementing MonteCarlo policy gradient algorithm (REINFORCE, [Williams, 1992])
# for learning an optimal graph for performing a bropadcast MPI collective operation.
#
# Optimal: assumed that reward depends on depth and type of comm. channel (by now).
#

# In[1]:


import gym
import gym_mpicolls
import numpy  as np

import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from torch.distributions import Categorical


#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

import sys
import time
import pdb


# In[2]:



# In[3]:


# Load environment and see spaces

# P = 8

M = (0, 0, 0, 0, 1, 1, 1, 1)  # P = 8
# M = (0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2) # P = 12
# M = (0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3) # P = 16



# In[4]:


class PolicyNetwork(nn.Module):
	
	def __init__(self, num_inputs, num_outputs, hidden_size):
	
		super(PolicyNetwork, self).__init__()
		
		self.num_outputs = num_outputs
		self.hidden1 = nn.Linear(num_inputs,  hidden_size)
		self.hidden2 = nn.Linear(hidden_size, hidden_size)
		self.output  = nn.Linear(hidden_size, num_outputs)
		
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




# In[5]:


# Agent

class Agent(object):
	
	def __init__ (self, env):
	
		self.gamma    = 1.00   # Almost Undiscounted
		self.alpha    = 0.001  # Learning rate (antes: 0.002)
		self.verbose  = False  # Verbosity
		
		self.policyNN = PolicyNetwork(P*P, P*P, 64)
		
		self.optimizer = torch.optim.Adam(self.policyNN.parameters(), lr=self.alpha)
		
		
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
		P_s = action.item() // P
		P_r = action.item() %  P
		
		
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
				
				
	def predict (self, s):

		# Transform into a Tensor
		state_tensor = torch.FloatTensor(s).view(1, -1)
	
		# Policy: next action
		action_probs = self.policyNN(state_tensor)
		
		# Sample
		action = action_probs.argmax()
		
		# Sender and receiver
		P_s = action.item() // P
		P_r = action.item() %  P
		
		return (P_s, P_r)




# In[6]:


def plot_graph(g, label):
	
	G = nx.DiGraph()
	G = nx.from_numpy_matrix(g)
	nx.draw_networkx(G, arrows=True, with_labels=True, node_color='w', label=label) #, layout='tree')
	
	plt.axis('off')



# TBD:
def generate_hosts_files(params):
	
	hosts_cur_name = params["Output"]["hosts_list"]
	
	hosts_list = list()

	# Write the host file
	hosts_f_name  = params["Output"]["hosts_file"]
	hosts_file = open(hosts_f_name,"w+")
	with open(hosts_cur_name, 'r') as h_file:
		for host in h_file:
			hosts_file.write(host.rstrip() + " " + "slots=24\n")
			hosts_list.append(host.rstrip())
	h_file.close()
	
	hosts_file.close()

	
	# Write the rank file (TODO)
	rank_f_name = params["Output"]["rank_file"]
	rank_file  = open(rank_f_name,"w+")

	#for p in range(P):
	#    'rank {}={} slot=0-5'.format(p, host, "0-5")
	rank_file.write('rank {}={} slot={}\n'.format(0, hosts_list[0], "0-5"))
	rank_file.write('rank {}={} slot={}\n'.format(1, hosts_list[1], "0-5"))
	rank_file.write('rank {}={} slot={}\n'.format(2, hosts_list[0], "6-11"))
	rank_file.write('rank {}={} slot={}\n'.format(3, hosts_list[1], "6-11"))
	rank_file.write('rank {}={} slot={}\n'.format(4, hosts_list[0], "12-17"))
	rank_file.write('rank {}={} slot={}\n'.format(5, hosts_list[1], "12-17"))
	rank_file.write('rank {}={} slot={}\n'.format(6, hosts_list[0], "18-23"))
	rank_file.write('rank {}={} slot={}\n'.format(7, hosts_list[1], "18-23"))

	rank_file.close()
				
				
	# Names for files
    # params["hosts_file"] = hosts_f_name
	# params["rank_file"] = rank_f_name
				
				
	return



# MAIN


import json

""" config = {
	
	"article": [
	
	{
	"id":"01",
	"language": "JSON",
	"edition": "first",
	"author": "Derrick Mwiti"
	},
	
	{
	"id":"02",
	"language": "Python",
	"edition": "second",
	"author": "Derrick Mwiti"
	}
	],
	
	"blog": [
	
	{
	"name": "Datacamp",
	"URL":"datacamp.com"
	}
	]
	
	} """

with open('config.json', 'r') as js:
	config = json.load(js)
	print(config)




a_params = config["Agent"]
print(a_params)


NO_EPISODES = a_params["n_episodes"]
interval = (NO_EPISODES // 20)


# M   = a_params["M"] # int(sys.argv[1])

P   = a_params["P"] # int(sys.argv[2])
# net = sys.argv[3]
# m   = int(sys.argv[4])

IMB     = config["Benchmark"]["exec"]
IMBOPT  = config["Benchmark"]["opts"]

graph_file  = config["Output"]["graph_file"]
hosts_file  = config["Output"]["hosts_file"]
output_file = config["Output"]["output_file"]

# print('Parameters: (M, P, net, m): ', M, P, net, m)

"""
params["P"] = P
params["M"] = M
params["m"] = m
params["net"] = net

params["graph_file"]  = graph_file
params["output_file"] = output_file
params["hosts_file"]  = hosts_file
params["rank_file"]   = ""
"""

generate_hosts_files(config)


# TBD: las siguientes lineas pueden sobrar:
# OPTIONS = IMB + " " + IMBOPT
# EXEC = OPTIONS.split()
# params["exec_command"] = EXEC




env = gym.make('MPIColls-v0', params=config["Environment"])

print("Observation space: ", env.observation_space)
print("Action space: ",      env.action_space)

print(env.observation_space.sample())
print(env.action_space.sample())




#  REINFORCE algorithm: Policy Gradients Monte Carlo (On-Policy)


# OJO: quitar esto al final
# np.random.seed(seed=42)


agent = Agent(env)

J_history = [] # np.zeros(NO_EPISODES)
T_history = [] # np.zeros(NO_EPISODES)
# D_history = np.zeros(NO_EPISODES)
R_history = [] # np.zeros(NO_EPISODES)

start = time.time()


print("Starting ...", flush=True)


for episode in range(NO_EPISODES):
#for episode in range(100):
	
	terminal = False
	reward   = 0
	steps    = 0
				
	s = env.reset()
	agent.reset()
				
	# if episode == 10:
	#     pdb.set_trace()
	#     print("+++++  EPISODE  +++++ ", episode)

	while not terminal:

		a = agent.select_action(s)

		s_, r, terminal, info = env.step(a)

		agent.save_step(s, a, r)
	
		s = np.copy(s_)
	
	
	agent.render()
	
	# Learn Policy
	J = agent.learn()

	J_history.append(J)
	T_history.append(len(agent.saved_rewards))
	R_history.append(sum(agent.saved_rewards))

	# J_history[episode] = J
	# T_history[episode] = len(agent.saved_rewards)
	# D_history[episode] = np.max(np.max(s_, axis=0))
	# R_history[episode] = sum(agent.saved_rewards)
	
	g = np.copy(s_)
		
		
		
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



window = 10
smoothed_rewards = [np.mean(R_history[i-window:i+1]) if i > window
					else np.mean(R_history[:i+1]) for i in range(len(R_history))]
"""
	plt.figure(figsize=(12,8))
	plt.plot(R_history)
	plt.plot(smoothed_rewards)
	#plt.ylim(bottom=min(smoothed_rewards), top=max(smoothed_rewards) + 1)
	plt.ylim(bottom=-150, top=max(smoothed_rewards) + 1)
	plt.ylabel('Total Rewards')
	plt.xlabel('Episodes')
	plt.show()
	"""


# Plot cost function values: J_history
# print(J_history)

j = np.array(J_history)
t = np.array(T_history)
# d = D_history


# Reduce dimensionality
X_AXIS = 100

j = j.reshape((X_AXIS, -1))

j_max = j.max(axis=1)
j_min = j.min(axis=1)
j_mean = j.mean(axis=1)

t = t.reshape((X_AXIS, -1))
t = t.mean(axis=1)

arr_mean = []
for i in j_mean:
  arr_mean.append(i.item())


j_ndarr = np.empty(j.shape, dtype=float)
for i in range(0, j.shape[0]-1):
  for k in range(0, j.shape[1]-1):
    j_ndarr[i][k] = j[i][k].item()

j_std = np.std(j_ndarr, axis=1)


plt.figure(figsize=(12,8))
plt.axis('on')

plt.plot(np.arange(0, X_AXIS, 1), j_mean, color='blue', marker='.')

plt.fill_between(np.arange(0, X_AXIS, 1), arr_mean - j_std, arr_mean + j_std, color='blue', alpha=0.2)

plt.title("Cost function per Episode")
plt.xlabel('# Episode')
plt.ylabel('J')
# plt.ylim(0)
plt.show()

"""
plt.plot(np.arange(0, X_AXIS, 1), t, color='red', marker='.')
plt.title("#steps in Trajectory per Episode")
plt.xlabel('# Episode')
plt.ylabel('T')
plt.ylim(0)
plt.show()

plt.plot(np.arange(0, X_AXIS, 1), d, color='green', marker='.')
plt.title("Depth of tree per Episode")
plt.xlabel('# Episode')
plt.ylabel('Depth')
plt.ylim(0)
plt.show()
"""





# PREDICT
s = env.reset()
agent.reset()
terminal = False
T = 0
print("PREDICT (only as an example): ")
print(s)
rewards=[]
while not terminal:
	
	a = agent.predict(s)
	print(a)
	s_, r, terminal, info = env.step(a)
	rewards.append(r)
	s = np.copy(s_)
	# print(s)

	T = T + 1
	if (T == 100):
		break

print("Rewards: ", rewards)

print(s_)

"""
	g = s_
	plot_graph(g, "Monte Carlo Policy Gradients")
	"""
