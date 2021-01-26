#!/usr/bin/env python

## MonteCarlo Policy Gradients (REINFORCE)
#
# Implementing MonteCarlo policy gradient algorithm
# (REINFORCE, [Williams, 1992]) to learn a mapping for a
# given communication graph representing an application.

import numpy  as np

import decimal
import time
import pdb
import json
import os
import sys
sys.path.append('./Env')
sys.path.append('./Agent')
sys.path.append('./utils')

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from torch.distributions import Categorical, Multinomial

from environ    import MPIMapEnv
from agent      import Agent
from graph      import adjacency
from config     import read_config
from outputs    import Output



##########   MAIN   ##########

# # Set to GPU
# if torch.cuda.is_available():
# 	print("Training model in GPUs ... ")
# 	torch.cuda.set_device("cuda:0")

# Read config/params from JSON file
config = read_config()


######################
# Main Learning code:
######################

# Create Environment instance
env    = MPIMapEnv(params=config)

# Create Agent instance
agent  = Agent(env=env, params=config)

# Output data
output = Output(agent=agent, env=env, params=config)


start = time.time()

n_episodes = config["Hyperparameters"]["n_episodes"]

for episode in range(n_episodes):

	terminal = False

	s = env.reset()
	agent.reset()

	while not terminal:

		a = agent.select_action(s)

		s_, r, terminal, info = env.step(a)

		agent.save_step(s, a, r, info)

		s = s_

	# Learn policy
	result = agent.learn()

	# Show/Write partial results
	output.render(result)


end = time.time()
print("Wallclock time: ", end - start)
