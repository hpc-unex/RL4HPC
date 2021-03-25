import gym
import gym_mpimap
from stable_baselines import A2C
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.env_checker import check_env

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import os
import sys

sys.path.append('./utils')
from config import read_config

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class MyMonitorWrapper(gym.Wrapper):

    def __init__(self, env):

        super(MyMonitorWrapper, self).__init__(env)

        self.episode_length = 0
        self.episode_reward = []

        self.info['episode_reward'] = []

        self.env = env


    def reset (self):

        # print("Reseting ...")
        self.episode_length = 0
        self.episode_reward = []

        obs = self.env.reset()

        return obs


    def step(self, action):

        obs, reward, done, info = self.env.step(action)
        self.episode_length += 1
        self.episode_reward.append(reward)

        if done:
            self.info['episode_length'] = self.episode_length
            self.info['episode_reward'].append(sum(self.episode_reward))

        return obs, reward, done, self.info


    def render(self, mode='human'):

        self.env.render(mode)

        fig = plt.figure("Reward")
        x = np.arange(len(self.info['episode_reward']))
        plt.plot(x, self.info['episode_reward'])
        plt.xlabel('Number of Timesteps')
        plt.ylabel('Rewards')
        plt.title("Reward " + " Smoothed")
        plt.show()



config = read_config()
env = MyMonitorWrapper(gym.make('MPIMap-v0', params=config))

check_env(env)


model = A2C("MlpPolicy", env, learning_rate=0.001 , verbose=0)
model.learn(total_timesteps=150000)


obs = env.reset()
for i in range(env.P):
    action, _states = model.predict(obs, deterministic=True)
    print("action:", action)
    #print("_states:", _states)
    obs, reward, done, info = env.step(action)
    #env.render()
    #print("i:", obs, info)
    #if done:
      #obs = env.reset()
      # print(info)

env.render()
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
