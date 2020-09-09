import numpy as np
import subprocess


from self.self_reward import get_reward
# from .tLop.tLop_reward import get_reward
# from .mpi.mpi_reward   import get_reward



class MPICollsEnv(object):
	
	
	def __init__(self, params):
		super(MPICollsEnv, self).__init__()
		
		self.params = params
		print(self.params)

		# ACTION space
		#self.action_space = Edge(P=self.params["P"])
	
		# STATES space
		#self.observation_space = Matrix(P=self.params["P"])
		
		
		self.state = None
		self.P = self.params["P"]
		self.root = self.params["root"]
		
		self.M = np.arange(0, self.params["M"], 1)
	
		self.trajectory = 0
	
	
	
	# This function returns a reward that favors Linear Trees.
	def step (self, action):
		
		src, dst = action

		# Stage: last stage of sending/receiving of a process + 1:
		stage = np.maximum( np.amax(self.state[src,:]), np.amax(self.state[:,src]) ) + 1
		valid = True
		reward = 0
		done  = False
		
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
			# reward = 0
		else: # if (valid == True):
			self.state[src,dst] = stage
			reward = 0.0
			# reward = 1
			
			# done = True if all processes received (columns have a value != -1)
			#print(self.state)
			#print(np.max(self.state, axis=0))
			#print(np.min(np.max(self.state, axis=0)))
			if np.min(np.max(self.state, axis=0)) > 0:
				done = True
				# reward = np.max(np.max(self.state, axis=0))
				#print("Reward: ", reward)
				reward = get_reward(self.state, self.M, self.P)
				# reward = -1 * get_reward_tlop (self.state, self.M, self.params)
				# reward = -1 * get_reward_mpi (self.state, self.M, self.params, self.trajectory)
				# print("REWARDS: ", reward, flush=True)
				self.trajectory += 1

		return np.array(self.state), reward, done, {}


		
	def reset(self):

		# root = self.observation_space.root
		# P = self.observation_space.P
		# self.state = np.ones((P, P), dtype=np.int) * -1
		self.state = np.zeros((self.P, self.P), dtype=np.int)
		# print("RESET STATE: ", P, self.state)
		
		self.state[self.root, self.root] = 1
		
		return np.copy(self.state)

	
		
	def render(self):
		print(self.state)
	
	
	
	def close(self):
		pass
