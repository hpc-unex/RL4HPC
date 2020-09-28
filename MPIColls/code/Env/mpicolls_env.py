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
		r = math.sqrt(get_reward(state, params))

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

		self.state = None
		self.P = self.params["P"]
		self.root = self.params["root"]

		self.M = np.arange(0, self.params["M"], 1)

		self.rw_type = self.params["reward_type"]

		self.verbose       = params["verbose"]  # Verbosity
		self.verbosity_int = params["verbosity_interval"]

		# ACTION space
		self.action_space = self.P

		# STATES space
		self.observation_space = self.P

		self.t = 0


	# This function returns a reward that favors Linear Trees.
	def step (self, action):

		src, dst = action

		self.t += 1

		# Stage: last stage of sending/receiving of a process + 1:
		stage = np.maximum( np.amax(self.state[src,:]), np.amax(self.state[:,src]) ) + 1
		valid  = True
		reward = 0
		done   = False

		# Case 1: A process sends w/o previously receiving:
		recv = np.max(self.state, axis=0)  # Last stage of reception of each process
		if (recv[src] <= 0) or (recv[src] >= stage):
			valid = False

		# Case 2: A process receives an already received message:
		#         axis=0 is by COLUMNS
		if (recv[dst] > 0):
			valid = False

		# Case 3: send self:
		if (src == dst):
			valid = False

		if (not valid):
			reward = -1.0

		else: # valid == True
			self.state[src,dst] = stage
			reward = 0.0

			if np.min(np.max(self.state, axis=0)) > 0:
				done = True
				reward = get_reward(self.rw_type, self.state, self.params)

		return np.array(self.state), reward, done, {"valid": valid}



	def reset(self):

		self.state = np.zeros((self.P, self.P), dtype=np.int)
		self.state[self.root, self.root] = 1

		self.t = 0

		return np.copy(self.state)



	def render(self, episode):

		if self.verbose and not (episode % self.verbosity_int):
			print(self.state)
