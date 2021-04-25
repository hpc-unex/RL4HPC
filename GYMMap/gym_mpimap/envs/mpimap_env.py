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
		self.cap = self.graph["capacity"]

		self.state = np.zeros(self.P, dtype=int)
		#self.state = []
		self.stepn = 0

		self.info = {}
		self.action_space = spaces.Discrete(self.M)

		self.observation_space = spaces.Box(low=0, high=self.M, shape=(self.P,), dtype=int)

		self.currP = 0
		self.comms = adjacency(self.P, self.graph["comms"])

		self.mcomm = np.zeros((np.max(self.graph["comms"]["edges"], axis=0)+1), dtype=int)

		for i,j in self.graph["comms"]["edges"]:
			self.mcomm[i][j] = 1
			
		
		#Create dict with comms		
		row_, col_ = np.nonzero(self.mcomm)
		dict_comms = defaultdict(set)
		[dict_comms[delvt].add(pin) for delvt, pin in zip(row_, col_)]
		self.dcomm = dict_comms
		
		self.levels = self.calculate_levels()
		
		
		self.depth = self.calculate_depth()


	def step(self, action):

		# 1. Apply action:
		# self.observation_space[self.currP] = action
		self.state[self.currP] = action
		self.currP = self.currP + 1
		self.stepn += 1
		r = 0
		if self.stepn % 10000 == 0:
			print(self.stepn * 8)

		# 2. Finish episode?
		done = False
		if self.currP == self.P:

			done = True
			# 3: Compute reward
			r = self.__compute_reward(action, done)
		return [self.state, r, done, self.info]


	def reset(self):

		self.currP = 0
		self.state = np.zeros(self.P, dtype=int)
		self.cap = [2] * self.M
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
		
		
	def calculate_depth(self):
		
		n_d = np.zeros(self.P)
		copy_dcomm = self.dcomm.copy()
		end = False
		d = 4
		first = True
		
		while len(copy_dcomm) != 0:
			for i in range(0,self.P):
				all_outs = True				
				if i not in copy_dcomm.keys() and first == True:
					n_d[i] = d
				elif i in copy_dcomm.keys() and first == False:
					for j in copy_dcomm[i]:
						if j in copy_dcomm.keys():
							all_outs = False
							break;
							 
					if all_outs == True:
						copy_dcomm.pop(i)
						n_d[i] = d
			d=3 if first==True else d-1
			first = False
			
		return n_d

	def __compute_reward(self, actions, done):

		MAX_R = 12 * 16384 * 64


		over_sub = np.ones(self.M, dtype=int)
		#D = np.ones((self.P, self.P))
		r = []

		#count = list(self.state).count(actions)

		for i in range(0, self.P):			
			if i == 0:
				r.append(0)
			else:
				for j in range(0, 4):
					for z in self.levels[j]:
						if i == z[1]:
								
							if self.state[i] == self.state[z[0]]:
								r.append(self.depth[i] * self.cap[self.state[i]])
							else:
								pass								
			self.cap[self.state[i]]-=1 #if self.cap[self.state[i]]>0 else 0
			
#		for i in range(0, self.P):
#			self.cap[self.state[i]] -= 1
#			for j in range(0, self.P):
#				if i == 0 and j == 0:
#					r.append(0)
#				elif i in self.dcomm[j]:
#					if self.cap[self.state[i]] < 0:
#						r.append(-10)
#					elif self.cap[self.state[j]] >= 0 and self.state[i] == self.state[j]:
#						r.append(self.depth[i]*2)
#					elif self.cap[self.state[j]] < 0 and self.state[i] == self.state[j]:
#						r.append(-10)
#					elif self.cap[self.state[j]] >= 0 and self.state[i] != self.state[j]:
#						r.append(-(self.depth[i]))
#					elif self.cap[self.state[j]] < 0 and self.state[i] != self.state[j]:
#						r.append(0)
#					else:
#						print("ERROR"); exit()
#				else:
#					pass

		r = sum(r)
		print("STATE:", self.state)
		# Normalize
		#r = np.sum(np.multiply(np.divide(r,MAX_R),100))

		return r
