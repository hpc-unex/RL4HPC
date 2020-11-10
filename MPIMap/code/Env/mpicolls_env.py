import numpy as np
import math
import subprocess
import sys
sys.path.append('../utils')
from graph  import plot_graph



def get_reward (rw_type, state, params):

	if rw_type == "self":
		from self.self_reward import get_reward
		r = get_reward(state, params)

	elif rw_type == "tLop":
		from tLop.tLop_reward import get_reward
		# r = math.sqrt(get_reward(state, params))
		r = get_reward(state, params)

	elif rw_type == "mpi":
		from .mpi.mpi_reward import get_reward
		r = 0.0

	else:
		r = 0.0

	return r




class MPICollsEnv(object):

	def __init__(self, params):
		super(MPICollsEnv, self).__init__()

		self.params = params

		self.P       = self.params["P"]
		self.root    = self.params["root"]
		self.M       = self.params["M"]
		self.procs   = self.params["nodes_procs"] # Processors assigned to a node

		self.rw_type = self.params["reward_type"]

		self.verbose       = self.params["verbose"]  # Verbosity
		self.verbosity_int = self.params["verbosity_interval"]

		# ACTION and STATE space
		self.action_space      = self.P
		self.observation_space = self.P

		self.t     = 0
		self.valid = True

		self.ro = self.params["ro"]
		self.c  = self.params["c"]

		self.n_intra_history  = []
		self.n_inter_history  = []
		self.n_errors_history = []
		self.episode = 0

		# Communication matrix: It represents the communications of the
		#  application and it is an input to this algorithm.
		# TBD: read it from a file
		comms = np.zeros((self.P, self.P), dtype=np.int)
		comms[0,1] = 1
		comms[1,3] = 1
		comms[0,2] = 1
		comms[0,4] = 1
		comms[1,5] = 1
		comms[2,6] = 1
		comms[3,7] = 1
		comms[0,8] = 1
		comms[1,9] = 1
		comms[2,10] = 1
		comms[3,11] = 1
		comms[4,12] = 1
		comms[5,13] = 1
		comms[6,14] = 1
		comms[7,15] = 1
		self.state = comms



	def step (self, actions):

		self.episode += 1

		self.t  = len(actions)
		rewards = np.zeros(self.t, dtype=np.float)

		valid  = True
		r      = 0
		done   = False

		n_errors = 0
		n_intra  = 0
		n_inter  = 0

		unique_elements, counts_elements = np.unique(actions, return_counts=True)
		Q = self.P // self.M


		counts_elements = counts_elements - Q
		n_errors = (self.M - np.size(counts_elements)) * Q + np.sum(np.abs(counts_elements))
		"""
		nodes = np.zeros(self.M, dtype=np.int)
		for i,a in enumerate(actions):
			nodes[a] += 1
			if (nodes[a] > Q):
				n_errors += 1
				rewards[i] = -1.0
		"""

		for src in range(0, self.P):
			for dst in range(0, self.P):
				if self.state[src,dst] != 0:
					if actions[src] == actions[dst]:
						n_intra += 1
					else:
						n_inter += 1


		if n_errors > 0:
			valid = False
		else:
			valid = True

		rewards[-1] = -(self.ro * n_intra + (1 - self.ro) * n_inter + self.c * n_errors)

		done = True

		return self.state, rewards, done, {"valid": valid, "n_errors": n_errors, "n_intra": n_intra, "n_inter": n_inter}


	def reset(self):

		#######################
		# Different forms of representing
		# a state (with one-hot enconding):
		# 0) Simple (adjacency without one-hot encoding)
		# 1) By an adjacency matrix
		# 2) By a communication layers matrix
		# 3) By a list of communication between ranks
		#######################

		s_rep = self.params["state_rep"]

		if s_rep == "Simple":

			# 0) By a simple matrix:

			# Open MPI Bcast  (16)
			adj = np.zeros((self.P, self.P), dtype=np.int)
			adj[0,1] = 1
			adj[1,3] = 1
			adj[0,2] = 1
			adj[0,4] = 1
			adj[1,5] = 1
			adj[2,6] = 1
			adj[3,7] = 1
			adj[0,8] = 1
			adj[1,9] = 1
			adj[2,10] = 1
			adj[3,11] = 1
			adj[4,12] = 1
			adj[5,13] = 1
			adj[6,14] = 1
			adj[7,15] = 1

			comms = adj
			comms = comms.reshape(self.P, -1)

		elif s_rep == "Adjacency":

			# 1) By Adjacency matrix:

			# Open MPI Bcast  (16)
			adj = np.zeros((self.P, self.P), dtype=np.int)
			adj[0,1] = 1
			adj[1,3] = 1
			adj[0,2] = 1
			adj[0,4] = 1
			adj[1,5] = 1
			adj[2,6] = 1
			adj[3,7] = 1
			adj[0,8] = 1
			adj[1,9] = 1
			adj[2,10] = 1
			adj[3,11] = 1
			adj[4,12] = 1
			adj[5,13] = 1
			adj[6,14] = 1
			adj[7,15] = 1

			comms = np.eye(2)[adj]
			comms = comms.reshape(self.P, -1)

		elif s_rep == "Layers":
			# 2) Layers:

			# Open MPI Bcast  (16)
			comms_output = np.zeros((self.P, self.P), dtype=np.int)
			comms_output[0,0] = 1
			comms_output[0,1] = 2
			comms_output[1,3] = 3
			comms_output[0,2] = 3
			comms_output[0,4] = 4
			comms_output[1,5] = 4
			comms_output[2,6] = 4
			comms_output[3,7] = 4
			comms_output[0,8] = 5
			comms_output[1,9] = 5
			comms_output[2,10] = 5
			comms_output[3,11] = 5
			comms_output[4,12] = 5
			comms_output[5,13] = 5
			comms_output[6,14] = 5
			comms_output[7,15] = 5

			comms_input = np.zeros((self.P, self.P), dtype=np.int)
			comms_input[0,0] = 1
			comms_input[1,0] = 2
			comms_input[2,0] = 3
			comms_input[3,1] = 3
			comms_input[4,0] = 4
			comms_input[5,1] = 4
			comms_input[6,2] = 4
			comms_input[7,3] = 4
			comms_input[8,0] = 5
			comms_input[9,1] = 5
			comms_input[10,2] = 5
			comms_input[11,3] = 5
			comms_input[12,4] = 5
			comms_input[13,5] = 5
			comms_input[14,6] = 5
			comms_input[15,7] = 5

			comms = np.eye(self.P)[comms_output]
			comms = comms.reshape(self.P, -1)

		elif s_rep == "Edges":

			# 3) Edges
			edges = np.zeros((self.P,2), dtype=np.int)
			edges[0] = [0,0]
			edges[1] = [0,1]
			edges[2] = [1,3]
			edges[3] = [0,2]
			edges[4] = [1,5]
			edges[5] = [3,7]
			edges[6] = [2,6]
			edges[7] = [0,4]
			edges[8] = [7,15]
			edges[9] = [3,11]
			edges[10] = [5,13]
			edges[11] = [1,9]
			edges[12] = [6,14]
			edges[13] = [2,10]
			edges[14] = [4,12]
			edges[15] = [0,8]

			comms = np.eye(self.P)[edges]
			comms = comms.reshape(self.P, -1)

		else:
			print("ERROR: state representation unsupported.")

		# Return state
		self.state = comms

		self.t = 0
		self.valid = False

		return self.state


	def render(self):

		if self.verbose and not (self.episode % self.verbosity_int):
			None
