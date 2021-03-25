import gym
import numpy as np

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

		self.state = np.zeros(self.P, dtype=int)

		self.info = {}
		self.action_space = spaces.Discrete(self.M)

		self.observation_space = spaces.Box(low=0, high=self.M, shape=(self.P,), dtype=int)

		self.currP = 0
		self.comms = adjacency(self.P, self.graph["comms"])



	def step(self, action):

		# 1. Apply action:
		# self.observation_space[self.currP] = action
		self.state[self.currP] = action
		self.currP = self.currP + 1
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

		return self.state


	def render(self, mode):

		# print("Process ", self.currP-1, " to node: ", self.state[self.currP-1])
		print(self.state)
		# print("Reward: ", self.reward)
		# print(self.observation_space


	def __compute_reward(self, action, done):

		r = 0
		MAX_R = 12 * 16384 * 64

		if done:
			over_sub = np.ones(self.M, dtype=int)
			D = np.ones((self.P, self.P))

			for m in range(self.M):
				over_sub[m] = np.abs(np.count_nonzero(self.state == m) - (self.P / self.M))

			for p in range(self.P):
				m = self.state[p]
				v = over_sub[m]
				idx = np.where(self.state == m)
				D[p][idx] = v

			r = np.sum(np.multiply(self.comms, D))

			# Normalize
			r = -(r/MAX_R)*100

			# print("Result: ")
			# print(self.state)
			# print(over_sub)
			# print(D)
		else:
			os = np.count_nonzero(self.state == action)

			dup = os - (self.P / self.M)
			if dup > 0:
				r = -dup

		#print("reward: ", r)
		return r
