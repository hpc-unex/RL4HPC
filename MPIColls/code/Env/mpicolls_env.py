import numpy as np
import math
import subprocess


# from self.self_reward import get_reward
from tLop.tLop_reward import get_reward
# from .mpi.mpi_reward   import get_reward



class MPICollsEnv(object):


	def __init__(self, params):
		super(MPICollsEnv, self).__init__()

		self.params = params

		self.state = None
		self.P = self.params["P"]
		self.root = self.params["root"]

		self.M = np.arange(0, self.params["M"], 1)

		self.verbose       = params["verbose"]  # Verbosity
		self.verbosity_int = params["verbosity_interval"]

		# ACTION space
		self.action_space = self.P

		# STATES space
		self.observation_space = self.P

		self.experience = 0

		self.less_states = []
		self.less_reward = -np.inf



	# This function returns a reward that favors Linear Trees.
	def step (self, action):

		src, dst = action

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
			reward = -1
			# reward = 0.0
		else: # valid == True
			self.state[src,dst] = stage
			reward = 0.0
			# reward = 1

			if np.min(np.max(self.state, axis=0)) > 0:
				done = True
				reward = - math.sqrt(get_reward(self.state, self.params)) 

				if self.less_reward < reward:
					self.less_reward = reward
					# print(reward)
					# print(self.state)
					self.less_states.append(np.copy(self.state))

				self.experience += 1

		return np.array(self.state), reward, done, {"valid": valid}



	def reset(self):

		self.state = np.zeros((self.P, self.P), dtype=np.int)
		self.state[self.root, self.root] = 1

		self.experience = 0

		return np.copy(self.state)



	def render(self, episode):

		if self.verbose and not (episode % self.verbosity_int):
			print(self.state)
			# print(self.less_states[-1])
