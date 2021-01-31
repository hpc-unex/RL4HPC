import numpy as np
import torch
import math
import subprocess
import sys
sys.path.append('../utils')
sys.path.append('../PG')

from graph  import plot_graph
from graph  import adjacency



class MPIMapEnv(object):

	def __init__(self, params):
		super(MPIMapEnv, self).__init__()

		self.params = params

		# Communication graph
		self.graph   = params["Graph"]
		self.P       = self.graph["P"]
		self.root    = self.graph["root"]
		self.M       = self.graph["M"]
		self.comms   = adjacency(self.P, self.graph["comms"])

		# Configuration parameters
		self.config  = params["Config"]
		self.rw_type = self.config["reward_type"]

		#
		# self.t = 0

		# Communication matrix: It represents the communications of the
		#  application and it is an input to this algorithm.
		# TBD: read it from a file
		# comms = np.zeros((self.P, self.P), dtype=np.int)
		# comms[0,0] = 1
		# comms[0,1] = 2
		# comms[1,3] = 3
		# comms[0,2] = 3
		# comms[0,4] = 4
		# comms[1,5] = 4
		# comms[2,6] = 4
		# comms[3,7] = 4
		"""
		comms[0,8] = 5
		comms[1,9] = 5
		comms[2,10] = 5
		comms[3,11] = 5
		comms[4,12] = 5
		comms[5,13] = 5
		comms[6,14] = 5
		comms[7,15] = 5
		comms[0,16] = 6
		comms[1,17] = 6
		comms[2,18] = 6
		comms[3,19] = 6
		comms[4,20] = 6
		comms[5,21] = 6
		comms[6,22] = 6
		comms[7,23] = 6
		comms[8,24] = 6
		comms[9,25] = 6
		comms[10,26] = 6
		comms[11,27] = 6
		comms[12,28] = 6
		comms[13,29] = 6
		comms[14,30] = 6
		comms[15,31] = 6
		"""

		# self.comms = comms

		# Optimization on GPUs
		if torch.cuda.is_available():
			self.device = torch.device('cuda')
		else:
			self.device = torch.device('cpu')


		# self.A = torch.FloatTensor(comms).detach().to(self.device)
		# self.A[self.A > 0] = self.m
		# print(self.A)

		"""
		self.D = torch.zeros((self.P, self.P)).to(self.device)
		for i in range(0, self.P):
			for j in range(0, self.P):
				if i == j:
					value = 100
				elif self.procs[i] == self.procs[j]:
					value = 1
				else:
					value = 10
				self.D[i,j] = value
		# print(self.D)

		# cost = torch.sum(self.A * self.D)
		# print("Coste inicial: ", cost)
		"""


	def __get_reward (self, state, actions):

		if self.rw_type == "self":
			from self.self_reward import get_reward
			r, info = get_reward(state, actions, self.params)

		elif self.rw_type == "tLop":
			from tLop.tLop_reward import get_reward
			r, info = get_reward(state, actions, self.params)

		elif self.rw_type == "mpi":
			from .mpi.mpi_reward import get_reward
			r = 0.0
			info = {}

		else:
			r = 0.0
			info = {}

		return r, info



	def step (self, actions, show=False):

		r, info = self.__get_reward(self.comms, actions)

		# self.baseline = self.baseline + (1.0 / self.episode) * (r - self.baseline)


		self.t  = len(actions)
		rewards = np.zeros(self.t, dtype=np.float)
		rewards[-1] = r

		"""
		# TEMPORAL
		Q = self.P // self.M

		errors = 0
		capacity = np.zeros(self.M, dtype=int)
		for i in range(0, self.P):
			node = actions[i]
			capacity[node] += 1
			if (capacity[node] > Q):
				rewards[i] = -1000
				errors += 1
		if errors == 0:
		 	rewards[-1] = r
		"""

		# reward = r

		done = True

		# print("[Env] state: ", self.state)
		# print("[Env] rewards: ", rewards)

		return self.state, rewards, done, info


	def reset(self):


		self.state = np.copy(self.comms)
		self.state[self.state > 0] = 1

		# print("[Env - reset()]: ")
		# print(self.state)

		self.t = 0
		self.valid = False

		return self.state

		#######################
		# Different forms of representing
		# a state (with one-hot enconding):
		# 0) Simple (adjacency without one-hot encoding)
		# 1) By an adjacency matrix
		# 2) By a communication layers matrix
		# 3) By a list of communication between ranks
		#######################


		""" LayersIO

		comms_output = np.zeros((self.P, self.P), dtype=np.int)
		comms_output[0,0] = 1
		comms_output[0,1] = 2
		comms_output[1,3] = 3
		comms_output[0,2] = 3
		comms_output[0,4] = 4
		comms_output[1,5] = 4
		comms_output[2,6] = 4
		comms_output[3,7] = 4

		comms_input = np.zeros((self.P, self.P), dtype=np.int)
		comms_input[0,0] = 1
		comms_input[1,0] = 2
		comms_input[2,0] = 3
		comms_input[3,1] = 3
		comms_input[4,0] = 4
		comms_input[5,1] = 4
		comms_input[6,2] = 4
		comms_input[7,3] = 4

		comms_o = np.eye(self.P)[comms_output]
		comms_o = comms_o.reshape(self.P, -1)

		# print("B ", comms_o.shape)

		comms_i = np.eye(self.P)[comms_input]
		comms_i = comms_i.reshape(self.P, -1)

		comms = np.concatenate((comms_o, comms_i), axis = 1)

		self.state = comms

		self.t = 0
		self.valid = False

		return self.state
		"""


		# s_rep = self.params["state_rep"]
		#
		# if s_rep == "Adjacency":
		#
		# 	# 0) By a simple matrix:
		#
		# 	# Open MPI Bcast  (16)
		# 	adj = np.zeros((self.P, self.P), dtype=np.int)
		# 	adj[0,1] = 1
		# 	adj[1,3] = 1
		# 	adj[0,2] = 1
		# 	adj[0,4] = 1
		# 	adj[1,5] = 1
		# 	adj[2,6] = 1
		# 	adj[3,7] = 1
		# 	adj[0,8] = 1
		# 	adj[1,9] = 1
		# 	adj[2,10] = 1
		# 	adj[3,11] = 1
		# 	adj[4,12] = 1
		# 	adj[5,13] = 1
		# 	adj[6,14] = 1
		# 	adj[7,15] = 1
		#
		# 	comms = adj
		# 	comms = comms.reshape(self.P, -1)
		#
		#
		# elif s_rep == "Layers":
		# 	# 2) Layers:
		#
		# 	# Open MPI Bcast  (16)
		# 	comms_output = np.zeros((self.P, self.P), dtype=np.int)
		# 	comms_output[0,0] = 1
		# 	comms_output[0,1] = 2
		# 	comms_output[1,3] = 3
		# 	comms_output[0,2] = 3
		# 	comms_output[0,4] = 4
		# 	comms_output[1,5] = 4
		# 	comms_output[2,6] = 4
		# 	comms_output[3,7] = 4
		# 	comms_output[0,8] = 5
		# 	comms_output[1,9] = 5
		# 	comms_output[2,10] = 5
		# 	comms_output[3,11] = 5
		# 	comms_output[4,12] = 5
		# 	comms_output[5,13] = 5
		# 	comms_output[6,14] = 5
		# 	comms_output[7,15] = 5
		#
		# 	comms = np.eye(self.P)[comms_output]
		# 	comms = comms.reshape(self.P, -1)
		#
		#
		# elif s_rep == "LayersIO":
		# 	# 2) Layers:
		#
		# 	# Open MPI Bcast  (16)
		# 	comms_output = np.zeros((self.P, self.P), dtype=np.int)
		# 	comms_output[0,0] = 1
		# 	comms_output[0,1] = 2
		# 	comms_output[1,3] = 3
		# 	comms_output[0,2] = 3
		# 	comms_output[0,4] = 4
		# 	comms_output[1,5] = 4
		# 	comms_output[2,6] = 4
		# 	comms_output[3,7] = 4
		# 	"""
		# 	comms_output[0,8] = 5
		# 	comms_output[1,9] = 5
		# 	comms_output[2,10] = 5
		# 	comms_output[3,11] = 5
		# 	comms_output[4,12] = 5
		# 	comms_output[5,13] = 5
		# 	comms_output[6,14] = 5
		# 	comms_output[7,15] = 5
		#
		# 	comms_output[0,16] = 6
		# 	comms_output[1,17] = 6
		# 	comms_output[2,18] = 6
		# 	comms_output[3,19] = 6
		# 	comms_output[4,20] = 6
		# 	comms_output[5,21] = 6
		# 	comms_output[6,22] = 6
		# 	comms_output[7,23] = 6
		# 	comms_output[8,24] = 6
		# 	comms_output[9,25] = 6
		# 	comms_output[10,26] = 6
		# 	comms_output[11,27] = 6
		# 	comms_output[12,28] = 6
		# 	comms_output[13,29] = 6
		# 	comms_output[14,30] = 6
		# 	comms_output[15,31] = 6
		# 	"""
		#
		# 	comms_input = np.zeros((self.P, self.P), dtype=np.int)
		# 	comms_input[0,0] = 1
		# 	comms_input[1,0] = 2
		# 	comms_input[2,0] = 3
		# 	comms_input[3,1] = 3
		# 	comms_input[4,0] = 4
		# 	comms_input[5,1] = 4
		# 	comms_input[6,2] = 4
		# 	comms_input[7,3] = 4
		# 	"""
		# 	comms_input[8,0] = 5
		# 	comms_input[9,1] = 5
		# 	comms_input[10,2] = 5
		# 	comms_input[11,3] = 5
		# 	comms_input[12,4] = 5
		# 	comms_input[13,5] = 5
		# 	comms_input[14,6] = 5
		# 	comms_input[15,7] = 5
		#
		# 	comms_input[16,0] = 6
		# 	comms_input[17,1] = 6
		# 	comms_input[18,2] = 6
		# 	comms_input[19,3] = 6
		# 	comms_input[20,4] = 6
		# 	comms_input[21,5] = 6
		# 	comms_input[22,6] = 6
		# 	comms_input[23,7] = 6
		# 	comms_input[24,8] = 6
		# 	comms_input[25,9] = 6
		# 	comms_input[26,10] = 6
		# 	comms_input[27,11] = 6
		# 	comms_input[28,12] = 6
		# 	comms_input[29,13] = 6
		# 	comms_input[30,14] = 6
		# 	comms_input[31,15] = 6
		# 	"""
		#
		# 	# print("A ", comms_output.shape)
		#
		# 	comms_o = np.eye(self.P)[comms_output]
		# 	comms_o = comms_o.reshape(self.P, -1)
		#
		# 	# print("B ", comms_o.shape)
		#
		# 	comms_i = np.eye(self.P)[comms_input]
		# 	comms_i = comms_i.reshape(self.P, -1)
		#
		# 	comms = np.concatenate((comms_o, comms_i), axis = 1)
		#
		# 	# print("C ", comms.shape)
		#
		#
		# elif s_rep == "Edges":
		#
		# 	# 3) Edges
		# 	edges = np.zeros((self.P,2), dtype=np.int)
		# 	edges[0] = [0,0]
		# 	edges[1] = [0,1]
		# 	edges[2] = [1,3]
		# 	edges[3] = [0,2]
		# 	edges[4] = [1,5]
		# 	edges[5] = [3,7]
		# 	edges[6] = [2,6]
		# 	edges[7] = [0,4]
		# 	edges[8] = [7,15]
		# 	edges[9] = [3,11]
		# 	edges[10] = [5,13]
		# 	edges[11] = [1,9]
		# 	edges[12] = [6,14]
		# 	edges[13] = [2,10]
		# 	edges[14] = [4,12]
		# 	edges[15] = [0,8]
		#
		# 	comms = np.eye(self.P)[edges]
		# 	comms = comms.reshape(self.P, -1)
		#
		# else:
		# 	print("ERROR: state representation unsupported.")

		# Return state
		# self.state = comms
		#
		# self.t = 0
		# self.valid = False
		#
		# return self.state
