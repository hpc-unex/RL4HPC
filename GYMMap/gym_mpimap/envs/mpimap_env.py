import gym
import numpy as np
from collections import defaultdict

from gym import error, spaces, utils
from gym.utils import seeding
from config import adjacency


class MPIMap(gym.Env):
	metadata = {'render.modes': ['human']}


	def __init__(self, params):
	
		self.graph = params["Graph"]
		self.P = self.graph["P"]
		self.M = self.graph["M"]
		self.m = self.graph["m"]
		self.cap_int = int(self.graph["cap"])
		self.cap = self.graph["capacity"]

		self.state = np.zeros(self.P, dtype=int)
		#self.state = []
		self.stepn = 0

		self.info = {}
		self.action_space = spaces.Discrete(self.M)

		self.observation_space = spaces.Box(low=0, high=self.M, shape=(self.P,), dtype=int)

		self.graph["comms"], self.depth, self.msgs = self.binomial_broadcast()

		self.currP = 0
		self.comms = adjacency(self.P, self.graph["comms"], self.msgs)

		#self.mcomm = np.zeros((np.max(self.graph["comms"]["edges"], axis=0)+1), dtype=int)
		self.mcomm = np.zeros((np.max(self.graph["comms"], axis=0)+1), dtype=int)

		#for i,j in self.graph["comms"]["edges"]:
		for i,j in self.graph["comms"]:
			self.mcomm[i][j] = 1
			
		
		#Create dict with comms		
		row_, col_ = np.nonzero(self.mcomm)
		dict_comms = defaultdict(set)
		[dict_comms[delvt].add(pin) for delvt, pin in zip(row_, col_)]
		self.dcomm = dict_comms
		
		self.levels = self.calculate_levels()
		
		#self.depth = self.calculate_depth()


	def step(self, action):

		# 1. Apply action:
		# self.observation_space[self.currP] = action
		self.state[self.currP] = action
		self.currP = self.currP + 1
		self.stepn += 1
		r = 0
		#if self.stepn % 10000 == 0:
		#	print(self.stepn * 4)

		# 2. Finish episode?
		done = False
		if self.currP == self.P:

			done = True
			# 3: Compute reward
			r = self.__compute_reward(action, done)
			print(self.stepn, "State:", self.state, "R:", r)
		return [self.state, r, done, self.info]


	def reset(self):

		self.currP = 0
		self.state = np.zeros(self.P, dtype=int)
		self.cap = [self.cap_int] * self.M
		return self.state


	def render(self, mode):

		# print("Process ", self.currP-1, " to node: ", self.state[self.currP-1])
		print(self.state)
		# print("Reward: ", self.reward)
		# print(self.observation_space
		
	def calculate_levels(self):
	
		i = 0
		d = {}
		for row in self.mcomm:
			for col in range(0, len(row)):
				if self.mcomm[i][col] == 1:
					d.setdefault(i, []).append([i,col])
			i += 1
		self.rows = i-1
		return d
	
	def binomial_broadcast(self):
		order=5
		nodes = sum([2**k for k in range(order)])

		lista_com = []
		depth = [0]
		msgs = []
		for i in range(int(np.ceil(np.log2(nodes)))):
			n_com = 2**i-1
			for idnode in range(nodes+1): # [0,nodes+1)
				if idnode+2**i <= nodes:
					lista_com.append([idnode, idnode+2**i])
					depth.append(i)
					msgs.append(65536) #Tamaño de mensaje
				if n_com == 0: break
				else: n_com-= 1
		return lista_com, depth, msgs

	def __compute_reward(self, actions, done):

		MAX_R = 12 * 16384 * 64


		over_sub = np.ones(self.M, dtype=int)
		#D = np.ones((self.P, self.P))
		r = []

		for i in range(0, self.P):			
			if i == 0:
				r.append(0)
			else:
				for j in range(0, np.max([a for a in self.levels.keys()])+1):
					for z in self.levels[j]:
						if i == z[1]:
							if self.state[i] == self.state[z[0]]:
								r.append(self.depth[i])
							else:
								pass	
												
			self.cap[self.state[i]]-=1 #if self.cap[self.state[i]]>0 else 0
		r = sum(r)

		for i in self.cap:
			if i < 0:
				return 0
		if r > 64:
			return 1
		return r**1/64**1
		# Normalize
		#r = np.sum(np.multiply(np.divide(r,MAX_R),100))

		#return r
