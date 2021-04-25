import gym
import gym_mpimap
from stable_baselines3 import A2C
#from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy


import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import os
import sys

sys.path.append('./utils')
from config import read_config
from plots import plot

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class MyMonitorWrapper(gym.Wrapper):

    def __init__(self, env):

        super(MyMonitorWrapper, self).__init__(env)

        self.episode_length = 0
        self.episode_reward = []

        self.info['episode_reward'] = []
        self.actions = []
        self.env = env


    def reset (self):

        # print("Reseting ...")
        self.episode_length = 0
        self.episode_reward = []
        self.actions = []
        obs = self.env.reset()

        return obs


    def step(self, action):

        self.episode_length += 1

        obs, reward, done, info = self.env.step(action)

        if self.episode_length == 8:
          done = True
          
        self.episode_reward.append(reward)

        if done:
            self.info['episode_length'] = self.episode_length
            self.info['episode_reward'].append(sum(self.episode_reward))

        return obs, reward, done, self.info


    def render(self, mode='human'):

        self.env.render(mode)
        print("Creating output plot...")
        plot(self.info)


def make_env(env_id, rank, config=None, seed=0):
	"""
	Utility function for multiprocessed env.
	:param env_id: (str) the environment ID
	:param seed: (int) the inital seed for RNG
	:param rank: (int) index of the subprocess
	"""
	def _init():
	    env = MyMonitorWrapper(gym.make('MPIMap-v0', params=config))
	    # Important: use a different seed for each environment
	    env.seed(seed + rank)
	    return env
	    
	set_random_seed(seed)
	return _init

config = read_config()
n_procs = 8
#env = MyMonitorWrapper(gym.make('MPIMap-v0', params=config))
#env = DummyVecEnv([lambda: MyMonitorWrapper(gym.make('MPIMap-v0', params=config))])
env = SubprocVecEnv([make_env('MPIMap-v0', i, config) for i in range(n_procs)], start_method='fork')
#eval_env = MyMonitorWrapper(gym.make('MPIMap-v0', params=config))

#check_env(env)

rewards = []

time_start = time.time()
model = A2C("MlpPolicy", env, n_steps=8, learning_rate=0.001, gamma=1, verbose=0)
model.learn(total_timesteps=100000)
print("FUN")
exit()
obs = env.reset()
for i in range(8):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    #env.render()
    #print("i:", obs, info)
    #if done:
      #obs = env.reset()
      # print(info)
print(time.time() - time_start)
#print("Mean Rewards: ", np.mean(rewards))
#env.render()

print(info[0]['terminal_observation'])
print(info[1]['terminal_observation'])
print(info[2]['terminal_observation'])
print(info[3]['terminal_observation'])
env.close()

"""
for episode in range(10):

    env.reset()
    score = 0

    for _ in range(env.P):

        action = env.action_space.sample()

        s, r, done, info = env.step(action)

        score = score + r

    env.render()
    print("Score: ", score)

    env.close()
"""
