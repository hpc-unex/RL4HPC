#!/usr/bin/env python

## MonteCarlo Policy Gradients (REINFORCE)
#
# Implementing MonteCarlo policy gradient algorithm
# (REINFORCE, [Williams, 1992]) to learn a mapping for a
# given communication graph representing an application.

import numpy  as np

import time
import json
import os
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from torch.distributions import Categorical, Multinomial

from environ    import MPIMapEnv
from agent      import Agent
from graph      import adjacency



class Output(object):

	def __init__ (self, agent, env, params):

		# Communication graph
		self.graph      = params["Graph"]
		self.P          = self.graph["P"]
		self.root       = self.graph["root"]
		self.M          = self.graph["M"]
		self.node_names = self.graph["node_names"]
		self.comms      = adjacency(self.P, self.graph["comms"])

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

		# Output config.
		self.output = params["Output"]
		output_file = self.output["output_file"]

		# Other variables
		self.J_history   = []
		self.R_history   = []
		self.T_history   = []

		self.agent = agent
		self.env   = env

		# Write header to file
		self.f = open(output_file, "w")

		self.f.write("#P: "           + str(self.P)             + "\n")
		self.f.write("#M: "           + str(self.M)             + "\n")
		self.f.write("#alpha: "       + str(self.alpha)         + "\n")
		self.f.write("#gamma: "       + str(self.gamma)         + "\n")
		self.f.write("#n_episodes: "  + str(self.n_episodes)    + "\n")
		self.f.write("#K: "           + str(self.K)             + "\n")
		self.f.write("#Baseline: "    + str(self.baseline_type) + "\n")

		self.f.write("#Node names: "  + str(self.node_names)    + "\n")
		# f.write("#Processors/Node: " + str(penv["nodes_procs"]) + "\n")
		self.f.write("#Reward_type: " + str(self.rw_type)       + "\n")
		# f.write("#StateRep: " + str(penv["state_rep"]) + "\n")
		self.f.write("#StartTime: "   + str(time.time())        + "\n")

		try:
			slurm_job_id   = os.environ['SLURM_JOB_ID']
			slurm_job_id   = "slurm-" + slurm_job_id + ".txt"
			slurm_nodelist = os.environ['SLURM_NODELIST']
		except:
			slurm_job_id   = '0'
			slurm_nodelist = 'local'

		self.f.write("#slurm file: "  + str(slurm_job_id)    + "\n")
		self.f.write("#node list: "   + str(slurm_nodelist)  + "\n")

		self.f.write("# \n")

		self.f.write("#  e   \t   J   \t  time  \t   T   \t reward \t baseline \t Actions \n")
		self.f.write("#----- \t ----- \t ------ \t ----- \t ------ \t -------- \t ------- \n")


	def __output_screen (self, info):

		n_episode     = info["Episode"]
		J             = info["J"]
		T             = info["T"]
		reward        = info["Reward"]
		baseline      = info["Baseline"]
		actions       = info["Actions"]
		returns       = info["Reward_seq"]
		discounted_rw = info["Discounted rw"]
		logprobs      = info["Logprobs"]

		self.J_history.append(J)
		self.R_history.append(reward)
		self.T_history.append(T)

		if self.verbose and not (n_episode % self.verbosity_int):

			start = n_episode - self.verbosity_int
			end   = n_episode
			n     = end - start

			if n != 0:

				costs = np.array(self.J_history)

				print("\n Episode ", n_episode," of ", self.n_episodes)
				print("--------------------------------------")

				print("Loss:                         ", sum(self.J_history[start:end]) / n)
				print("Acc. Loss (mean/std/max/min): ", costs.mean(), costs.std(), costs.max(), costs.min())

				print("Reward:      ", sum(self.R_history[start:end]) / n)
				print("Avg. Reward: ", sum(self.R_history) / len(self.R_history))
				print("T:           ", sum(self.T_history[start:end]) / n)

				print("Return:        ", returns)
				print("Discounted rw: ", discounted_rw)
				print("Actions:       ", actions)
				print("Log probs:     ", logprobs)

				# Greedy trajectory:
				a, action_probs, r = self.agent.predict_trajectory()
				print("===>> Greedy Trajectory: ")
				print("Actions: ", a)
				print("Logits:  ", action_probs)
				print("Rewards: ", r)

				print(flush=True)

		return


	def __output_file (self, info):

		n_episode = info["Episode"]
		J         = info["J"]
		T         = info["T"]
		reward    = info["Reward"]
		baseline  = info["Baseline"]
		actions   = info["Actions"]

		self.f.write(str(n_episode)   + " # " +
					 str(J)           + " # " +
					 str(time.time()) + " # " +
					 str(T)           + " # " +
					 str(reward)      + " # " +
					 str(baseline)    + " # " +
					 str(actions)     + "\n")

		self.f.flush()


	def render(self, info):

		n_episode = info["Episode"]

		self.__output_file(info)
		self.__output_screen(info)

		if n_episode == self.n_episodes:
			self.f.close()
